"""
Training Script: HR Simultaneous Inference (component-R² validation)
=====================================================================
Same data / model / optimizer as HR_dyn/train_simul_HR.py.  The ONLY
difference is the validation / early-stopping criterion:

    Instead of MSE(y_true, y_pred), we compare the learned MLPs against
    the analytic HR components on the test set:
        - node_fnc_x(x)  vs.  y - x^3 + 3 x^2 - z + 3.24
        - node_fnc_y(x)  vs.  1 - 5 x^2 - y
        - node_fnc_z(x)  vs.  0.004 * (4 (x + 1.6) - z)
        - soft_weights * msg_fnc([x_i, x_j])
              vs.  Aij_true * 0.15 * (2 - x_i) / (1 + exp(-10 (x_j - 1)))
    (See HR_dyn/reference_codes/knstruc.ipynb -> self_message_error and
     tools.py -> get_messages_with_trueAij)

Early-stop metric = mean of (1 - R²) across the 4 components (smaller = better).
Lower is better.
"""

import os
import argparse
import time
import math
import copy
import pickle
import json
import glob
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

import sys
# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'HR_dyn'))

from model import (
    tau_soft_regu_deriv_MP,
    GraphDataset,
    get_edge_index,
    calculate_auc,
    count_parameters,
    uniform_subsample_indices,
    evaluate_on_set,
)


def one_minus_corr2(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    if a.size == 0 or np.std(a) == 0 or np.std(b) == 0:
        return float('nan')
    c = np.corrcoef(a, b)[0, 1]
    return float(1.0 - c ** 2) if np.isfinite(c) else float('nan')


NET_NAME = 'Celegans'
T_TOTAL = 500
DT_DATA = 0.01
DIMENSION = 3


def float_to_tag(value):
    return format(value, "g").replace(".", "")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in {"true", "1", "yes"}:
        return True
    if v.lower() in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v}")


def _unwrap(m):
    return getattr(m, '_orig_mod', m)


@torch.no_grad()
def compute_component_metrics(Dyn, X_set, y_set, edge_index, objectAij, device):
    """Evaluate component R² on test set.

    Self components: same as knstruc (compare node_fnc_* with HR equations).
    Message component: compare
        Aij_true * G_true(x_i, x_j)  vs  soft_weights * msg_fnc(x_i, x_j)
    over all edges × all test samples.
    """
    core = _unwrap(Dyn)
    core.eval()

    overall = evaluate_on_set(Dyn, X_set, y_set, edge_index, device)

    T = X_set.shape[0]
    N = X_set.shape[1]
    ei = edge_index.to(device)
    src_idx = ei[0]
    tgt_idx = ei[1]

    # self dynamics on node-level states
    x_flat = X_set.to(device).reshape(T * N, DIMENSION)
    sx = core.node_fnc_x(x_flat).reshape(-1).cpu().numpy()
    sy = core.node_fnc_y(x_flat).reshape(-1).cpu().numpy()
    sz = core.node_fnc_z(x_flat).reshape(-1).cpu().numpy()
    xa = x_flat.cpu().numpy()
    sx_true = xa[:, 1] - xa[:, 0] ** 3 + 3 * xa[:, 0] ** 2 - xa[:, 2] + 3.24
    sy_true = 1.0 - 5.0 * xa[:, 0] ** 2 - xa[:, 1]
    sz_true = 0.004 * (4.0 * (xa[:, 0] + 1.6) - xa[:, 2])

    # Message: Tmp = [x_target_dim0, x_source_dim0] per edge (matches model.message)
    X_dev = X_set.to(device)
    xi_all = X_dev[:, tgt_idx, 0]        # (T, E)  target first-dim
    xj_all = X_dev[:, src_idx, 0]        # (T, E)  source first-dim
    xi_flat = xi_all.reshape(-1)
    xj_flat = xj_all.reshape(-1)
    tmp_in = torch.stack([xi_flat, xj_flat], dim=1)
    G_pred_flat = core.msg_fnc(tmp_in).reshape(-1)    # (T*E,)

    # Inferred soft weights (per edge, shape [E])
    soft_w = F.softmax(core.weights / core.tau, dim=1)[:, 0]   # (E,)
    # Tile soft_w across T test samples: (T, E) -> (T*E,)
    soft_w_tiled = soft_w.unsqueeze(0).expand(T, -1).reshape(-1)
    msg_pred_weighted = (soft_w_tiled * G_pred_flat).cpu().numpy()

    # True Aij per edge: objectAij.T flattened over off-diagonal, matching
    # the row-major order of edge_index = np.where(initialA) with initialA
    # all-ones minus diagonal.
    objA = np.asarray(objectAij)
    mask = np.ones_like(objA, dtype=bool)
    np.fill_diagonal(mask, 0)
    aij_true = objA.T[mask].astype(np.float32)           # (E,)
    xi_n = xi_flat.cpu().numpy()
    xj_n = xj_flat.cpu().numpy()
    G_true_flat = 0.15 * (2.0 - xi_n) / (1.0 + np.exp(-10.0 * (xj_n - 1.0)))
    aij_tiled = np.tile(aij_true, T)
    msg_true_weighted = aij_tiled * G_true_flat

    sx_err = one_minus_corr2(sx_true, sx)
    sy_err = one_minus_corr2(sy_true, sy)
    sz_err = one_minus_corr2(sz_true, sz)
    msg_err = one_minus_corr2(msg_true_weighted, msg_pred_weighted)

    comp_err = float(np.nanmean([sx_err, sy_err, sz_err, msg_err]))

    try:
        auc, auprc = calculate_auc(objectAij, core.weights)
    except Exception:
        auc, auprc = float('nan'), float('nan')

    return {
        'overall_r2': overall['r2'],
        'node_r2_mean': overall['node_r2_mean'],
        'self_x_err': float(sx_err),
        'self_y_err': float(sy_err),
        'self_z_err': float(sz_err),
        'msg_err': float(msg_err),
        'component_mean_err': comp_err,
        'AUC': float(auc),
        'AUPRC': float(auprc),
    }


def main():
    parser = argparse.ArgumentParser(
        description="HR simultaneous inference with component-R² validation")
    parser.add_argument('--num_train_samples', type=int, default=500)
    parser.add_argument('--num_nodes_keep', type=int, default=279)
    parser.add_argument('--data_seed', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--lam', type=float, default=0.001)
    parser.add_argument('--tau_update', type=float, default=0.999)
    parser.add_argument('--hidden', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_ratio', type=float, default=0.1)
    parser.add_argument('--use_early_stop', type=str2bool, default=False)
    parser.add_argument('--early_stop_patience', type=int, default=100)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--outdir', type=str, default='')
    args = parser.parse_args()

    torch.cuda.set_device(args.device_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    nodes_num = args.num_nodes_keep
    dt_str = float_to_tag(DT_DATA)
    base = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base, 'data_new')
    series_pattern = os.path.join(
        data_dir,
        f'Series_N{nodes_num}_{NET_NAME}_T{T_TOTAL}_dt{dt_str}_gc*_d*_seed{args.data_seed}.pickle')
    series_matches = sorted(glob.glob(series_pattern))
    if len(series_matches) != 1:
        raise FileNotFoundError(
            f'Expected exactly one series file for pattern {series_pattern}, found {len(series_matches)}: {series_matches}')
    series_path = series_matches[0]
    match = re.search(r'_gc([^_]+)_d([^_]+)_seed', os.path.basename(series_path))
    if match is None:
        raise ValueError(f'Could not parse degree tag from {series_path}')
    gc_tag = match.group(1)
    degree_tag = match.group(2)
    actual_gc = float(gc_tag.replace('p', '.'))
    dxdt_path = os.path.join(
        data_dir,
        f'TrueDxdt_N{nodes_num}_{NET_NAME}_T{T_TOTAL}_dt{dt_str}_gc{gc_tag}_d{degree_tag}_seed{args.data_seed}.pickle')

    with open(series_path, 'rb') as f:
        objectAij, series = pickle.load(f)
    with open(dxdt_path, 'rb') as f:
        _, dxdt_all = pickle.load(f)

    print(f'Loaded series: {series.shape}, dxdt: {dxdt_all.shape}')
    print(f'Adjacency: {objectAij.shape}, nodes={nodes_num}, seed={args.seed}')

    series = series.reshape(-1, nodes_num, DIMENSION)
    dxdt_all = dxdt_all.reshape(-1, nodes_num, DIMENSION)

    if args.outdir:
        data_path = args.outdir
    else:
        data_path = (f'HR_dyn/results_simul_new/Nodes{nodes_num}_d{degree_tag}_gc{gc_tag}'
                     f'_Ntrain{args.num_train_samples}_hidden{args.hidden}_seed{args.seed}/')
    os.makedirs(data_path, exist_ok=True)

    train_end = int(250 / DT_DATA)
    eval_end = int(400 / DT_DATA)

    series_train_full = series[:train_end]
    dxdt_train_full = dxdt_all[:train_end]

    VALTEST_SEED = 2024
    eval_pool_size = eval_end - train_end
    rng_valtest = np.random.RandomState(VALTEST_SEED)
    eval_indices = rng_valtest.choice(eval_pool_size, size=1000, replace=False)
    val_indices = np.sort(eval_indices[:500]) + train_end
    test_indices = np.sort(eval_indices[500:]) + train_end

    series_val = series[val_indices]; dxdt_val = dxdt_all[val_indices]
    series_test = series[test_indices]; dxdt_test = dxdt_all[test_indices]

    train_full_size = series_train_full.shape[0]
    if args.num_train_samples > 0 and args.num_train_samples < train_full_size:
        train_indices = uniform_subsample_indices(train_full_size, args.num_train_samples)
    else:
        train_indices = np.arange(train_full_size, dtype=np.int64)
    effective_num_train = train_indices.shape[0]

    series_train = series_train_full[train_indices]
    dxdt_train = dxdt_train_full[train_indices]

    print(f'Train: {series_train.shape[0]} points')
    print(f'Test:  {series_test.shape[0]} points')

    initialA = np.ones((nodes_num, nodes_num))
    np.fill_diagonal(initialA, 0)
    # edge_index convention (matches reference simul `train_truedxdt.py`):
    # edge_index[0] = HR-source, edge_index[1] = HR-target. initialA is
    # symmetric all-ones so np.where(initialA) == np.where(initialA.T); the
    # ordering and roles match the reference. `self.weights[k]` is learned to
    # approximate A[edge_index[1][k], edge_index[0][k]] (i.e. the HR adjacency
    # entry A[HR-target, HR-source]). When reconstructing a dense inferred
    # adjacency, use A_recon[edge_index[1], edge_index[0]] = softmax(weights)
    # — reading the weights back as A_recon[ei[0], ei[1]] would give A.T.
    edge_index = get_edge_index(initialA)

    X_train = torch.as_tensor(series_train.astype('float')).float()
    y_train = torch.as_tensor(dxdt_train.astype('float')).float()
    X_val = torch.as_tensor(series_val.astype('float')).float()
    y_val = torch.as_tensor(dxdt_val.astype('float')).float()
    X_test = torch.as_tensor(series_test.astype('float')).float()
    y_test = torch.as_tensor(dxdt_test.astype('float')).float()

    len_train = X_train.shape[0]
    batch = max(1, math.ceil(len_train * args.batch_ratio))
    print(f'Batch size: {batch}')

    train_dataset = GraphDataset(X_train, y_train, edge_index)
    trainloader = DataLoader(train_dataset, batch_size=batch, shuffle=False,
                             num_workers=4, pin_memory=True, persistent_workers=True)

    aggr = 'add'
    n_f = DIMENSION
    msg_dim = 1
    edge_num = edge_index.shape[1]
    tau = 1.0

    Dyn = tau_soft_regu_deriv_MP(
        n_f, msg_dim, ndim=DIMENSION, delt_t=DT_DATA,
        edge_num=edge_num, tau=tau, lam=args.lam,
        hidden=args.hidden, aggr=aggr
    ).to(device)
    Dyn = torch.compile(Dyn)
    print(f'Model params: {count_parameters(Dyn):,}')

    opt = torch.optim.AdamW(Dyn.parameters(), lr=args.lr, weight_decay=1e-8,
                            betas=(0.9, 0.999), eps=1e-8)
    batch_per_epoch = math.ceil(len(trainloader)) * 2
    sched = OneCycleLR(opt, max_lr=args.lr, steps_per_epoch=batch_per_epoch,
                       epochs=args.epochs, final_div_factor=1e5)

    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    VAL_EVERY = 10

    loss_over_time = []
    val_metric_over_time = []  # component_mean_err
    val_overall_r2_hist = []
    AUC_over_time = []
    AUPRC_over_time = []
    structure_weights = []
    best_val_metric = float('inf')
    best_state = None
    best_snapshot = None
    epochs_no_improve = 0

    start_time = time.time()

    for epoch in tqdm(range(args.epochs)):
        Dyn.train()
        total_loss = 0.0
        num_items = 0
        i = 0
        while i < batch_per_epoch:
            for ginput in trainloader:
                ginput = ginput.to(device)
                if i >= batch_per_epoch:
                    break
                opt.zero_grad()
                with torch.amp.autocast('cuda', enabled=use_amp):
                    loss = Dyn.loss(ginput, square=False)
                bsz = int(ginput.batch[-1] + 1)
                scaler.scale(loss / bsz).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                total_loss += loss.item()
                i += 1
                num_items += bsz

        tau *= args.tau_update
        _unwrap(Dyn).update_tau(tau)

        cur_loss = total_loss / num_items
        loss_over_time.append(cur_loss)

        cur_val_metric = val_metric_over_time[-1] if val_metric_over_time else float('inf')
        cur_metrics = None
        if (epoch + 1) % VAL_EVERY == 0 or epoch == 0:
            cur_metrics = compute_component_metrics(
                Dyn, X_test, y_test, edge_index, objectAij, device)
            cur_val_metric = cur_metrics['component_mean_err']
        val_metric_over_time.append(cur_val_metric)
        val_overall_r2_hist.append(cur_metrics['overall_r2'] if cur_metrics else
                                   (val_overall_r2_hist[-1] if val_overall_r2_hist else float('nan')))

        # per-epoch cheap stats
        cur_weights = _unwrap(Dyn).weights.cpu().detach().numpy()
        structure_weights.append(cur_weights)
        if cur_metrics is not None:
            AUC_over_time.append(cur_metrics['AUC'])
            AUPRC_over_time.append(cur_metrics['AUPRC'])
        else:
            AUC_over_time.append(AUC_over_time[-1] if AUC_over_time else float('nan'))
            AUPRC_over_time.append(AUPRC_over_time[-1] if AUPRC_over_time else float('nan'))

        if (epoch + 1) % 100 == 0:
            msg = f'Epoch {epoch+1}: train_loss={cur_loss:.4f}, comp_err={cur_val_metric:.4f}'
            if cur_metrics is not None:
                msg += (f', 1-corr²[self_x={cur_metrics["self_x_err"]:.3f}, '
                        f'self_y={cur_metrics["self_y_err"]:.3f}, '
                        f'self_z={cur_metrics["self_z_err"]:.3f}, '
                        f'msg={cur_metrics["msg_err"]:.3f}], '
                        f'overall_R²={cur_metrics["overall_r2"]:.3f}, '
                        f'AUC={cur_metrics["AUC"]:.3f}')
            print(msg)

        if cur_metrics is not None and cur_val_metric < best_val_metric:
            best_val_metric = cur_val_metric
            best_state = copy.deepcopy(_unwrap(Dyn).state_dict())
            best_snapshot = {
                'epoch': epoch + 1,
                'train_loss': cur_loss,
                **cur_metrics,
            }
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch + 1,
                'state_dict': best_state,
                'component_mean_err': best_val_metric,
                'metrics': cur_metrics,
                'train_loss': cur_loss,
                'config': vars(args),
            }, data_path + 'best_model.pt')
        elif cur_metrics is not None:
            epochs_no_improve += VAL_EVERY if epoch > 0 else 1

        if args.use_early_stop and args.early_stop_patience > 0 \
                and epochs_no_improve >= args.early_stop_patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

        should_save_periodic = (epoch + 1) % 100 == 0 and (
            args.use_early_stop or (epoch + 1) > 1000
        )

        if should_save_periodic:
            periodic_metrics = compute_component_metrics(
                Dyn, X_test, y_test, edge_index, objectAij, device)
            print(f'  [Periodic] Epoch {epoch+1}: overall_R²={periodic_metrics["overall_r2"]:.4f}, '
                  f'1-corr² self_x={periodic_metrics["self_x_err"]:.3f}, '
                  f'self_y={periodic_metrics["self_y_err"]:.3f}, '
                  f'self_z={periodic_metrics["self_z_err"]:.3f}, '
                  f'msg={periodic_metrics["msg_err"]:.3f}, '
                  f'AUC={periodic_metrics["AUC"]:.3f}')

            snap = {
                'epoch': epoch + 1,
                'train_loss': cur_loss,
                **periodic_metrics,
                'best_component_mean_err': float(best_val_metric),
            }
            with open(data_path + f'metrics_snapshot_e{epoch+1}.json', 'w') as f:
                json.dump(snap, f, indent=2, default=float)

            with open(data_path + f'weights_over_time_lam{args.lam}_e{epoch}.pkl', 'wb') as f:
                pickle.dump(structure_weights, f)

            Eva = pd.DataFrame({'AUC': AUC_over_time, 'AUPRC': AUPRC_over_time})
            Eva.to_csv(data_path + f'Eva_lam{args.lam}_e{epoch}.csv')

            torch.save(copy.deepcopy(_unwrap(Dyn).state_dict()),
                       data_path + f'recorded_model_e{epoch+1}.pt')

            structure_weights = []
            AUC_over_time = []
            AUPRC_over_time = []

    elapsed = time.time() - start_time
    print(f'\nTraining done in {elapsed:.1f}s')

    if best_state is not None:
        _unwrap(Dyn).load_state_dict(best_state)
        print(f'Loaded best model (component_mean_err={best_val_metric:.4f})')
    final_metrics = compute_component_metrics(
        Dyn, X_test, y_test, edge_index, objectAij, device)
    print(f'Test overall_R²={final_metrics["overall_r2"]:.4f}, '
          f'1-corr² self_x={final_metrics["self_x_err"]:.3f}, '
          f'self_y={final_metrics["self_y_err"]:.3f}, '
          f'self_z={final_metrics["self_z_err"]:.3f}, '
          f'msg={final_metrics["msg_err"]:.3f}')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(loss_over_time, label='train_loss')
    ax[0].legend(); ax[0].set_title('Train loss')
    ax[1].plot(val_metric_over_time, label='component_mean_err (test)')
    ax[1].legend(); ax[1].set_title('Component-mean error (early-stop metric)')
    plt.tight_layout()
    plt.savefig(data_path + 'loss_curves.png')
    plt.close()

    with open(data_path + f'loss_over_time_lam{args.lam}.pkl', 'wb') as f:
        pickle.dump({'train': loss_over_time,
                     'val_component_err': val_metric_over_time,
                     'val_overall_r2': val_overall_r2_hist}, f)

    results = {
        'nodes_num': int(nodes_num),
        'num_train_samples': int(effective_num_train),
        'seed': args.seed,
        'data_seed': args.data_seed,
        'degree_tag': degree_tag,
        'gc': actual_gc,
        'lam': args.lam,
        'tau_update': args.tau_update,
        'hidden': args.hidden,
        'epochs_trained': len(loss_over_time),
        'elapsed_seconds': elapsed,
        'best_component_mean_err': float(best_val_metric),
        'best_snapshot': best_snapshot,
        'final_metrics': final_metrics,
    }
    with open(data_path + 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)

    print(f'Results saved to {data_path}')


if __name__ == '__main__':
    main()
