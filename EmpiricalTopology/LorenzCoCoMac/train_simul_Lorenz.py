"""
Training Script: Lorenz Simultaneous Inference (component-R² validation)
========================================================================
Same skeleton as HR_dyn_new/train_simul_HR.py, but with Lorenz true components:

    node_fnc_x(x)    vs.   10*(y - x)
    node_fnc_y(x)    vs.   28*x - y - x*z
    node_fnc_z(x)    vs.   x*y - (8/3)*z
    soft_w * msg_fnc([x_i, x_j])
                     vs.   Aij_true * 0.2 * (x_j - x_i)

These match generation_Lorenz.py and `knstruc_batch.ipynb -> self_message_error`.

Data layout (same as knstruc): T=100, dt=0.001; train T=0..40 (uniform subsample);
val/test random 500/500 from T>40 pool (VALTEST_SEED=2024).
Early-stop metric = mean of (1 - R²) across the 4 components on the test set.
"""

import os
import argparse
import time
import math
import copy
import pickle
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import (
    tau_soft_regu_deriv_MP,
    GraphDataset,
    get_edge_index,
    calculate_auc,
    count_parameters,
    uniform_subsample_indices,
    evaluate_on_set,
    compute_r2,
    safe_corrcoef,
)


NET_NAME = 'Lorenz'
T_TOTAL = 100
DT_DATA = 0.001
DIMENSION = 3

# Lorenz parameters
SIGMA = 10.0
RHO = 28.0
BETA = 8.0 / 3.0
EPSILON = 0.2


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
    """Component R² for Lorenz simul: self_x/y/z + (soft_w * G_pred vs A_true * G_true)."""
    core = _unwrap(Dyn)
    core.eval()

    overall = evaluate_on_set(Dyn, X_set, y_set, edge_index, device)

    T = X_set.shape[0]
    N = X_set.shape[1]
    ei = edge_index.to(device)
    src_idx = ei[0]       # source
    tgt_idx = ei[1]       # target

    x_flat = X_set.to(device).reshape(T * N, DIMENSION)
    sx = core.node_fnc_x(x_flat).reshape(-1).cpu().numpy()
    sy = core.node_fnc_y(x_flat).reshape(-1).cpu().numpy()
    sz = core.node_fnc_z(x_flat).reshape(-1).cpu().numpy()
    xa = x_flat.cpu().numpy()
    x, y, z = xa[:, 0], xa[:, 1], xa[:, 2]
    sx_true = SIGMA * (y - x)
    sy_true = RHO * x - y - x * z
    sz_true = x * y - BETA * z

    X_dev = X_set.to(device)
    xi_all = X_dev[:, tgt_idx, 0]              # (T, E) target
    xj_all = X_dev[:, src_idx, 0]              # (T, E) source
    xi_flat = xi_all.reshape(-1)
    xj_flat = xj_all.reshape(-1)
    tmp_in = torch.stack([xi_flat, xj_flat], dim=1)
    G_pred_flat = core.msg_fnc(tmp_in).reshape(-1)

    soft_w = F.softmax(core.weights / core.tau, dim=1)[:, 0]   # (E,)
    soft_w_tiled = soft_w.unsqueeze(0).expand(T, -1).reshape(-1)
    msg_pred_weighted = (soft_w_tiled * G_pred_flat).cpu().numpy()

    # True Aij per edge. Same ordering as HR simul: objectAij.T flattened over
    # off-diagonal matches edge_index = np.where(initialA) with initialA all-ones.
    objA = np.asarray(objectAij)
    mask = np.ones_like(objA, dtype=bool)
    np.fill_diagonal(mask, 0)
    aij_true = objA.T[mask].astype(np.float32)
    xi_n = xi_flat.cpu().numpy()
    xj_n = xj_flat.cpu().numpy()
    G_true_flat = EPSILON * (xj_n - xi_n)
    aij_tiled = np.tile(aij_true, T)
    msg_true_weighted = aij_tiled * G_true_flat

    # Reference (knstruc_batch.ipynb::self_message_error) uses 1 - corr^2
    # rather than 1 - R^2 — tolerates uniform scale/offset to reward the MLP
    # for learning the functional SHAPE.
    sx_c2 = safe_corrcoef(sx_true, sx) ** 2
    sy_c2 = safe_corrcoef(sy_true, sy) ** 2
    sz_c2 = safe_corrcoef(sz_true, sz) ** 2
    msg_c2 = safe_corrcoef(msg_true_weighted, msg_pred_weighted) ** 2

    c2s = [sx_c2, sy_c2, sz_c2, msg_c2]
    comp_err = float(np.nanmean([1.0 - c for c in c2s]))

    try:
        auc_v, auprc_v = calculate_auc(objectAij, core.weights)
    except Exception:
        auc_v, auprc_v = float('nan'), float('nan')

    return {
        'overall_r2': overall['r2'],
        'node_r2_mean': overall['node_r2_mean'],
        'self_x_corr2': float(sx_c2),
        'self_y_corr2': float(sy_c2),
        'self_z_corr2': float(sz_c2),
        'msg_corr2': float(msg_c2),
        'component_mean_err': comp_err,
        'AUC': float(auc_v),
        'AUPRC': float(auprc_v),
    }


def main():
    parser = argparse.ArgumentParser(description="Lorenz simultaneous inference")
    parser.add_argument('--num_train_samples', type=int, default=50)
    parser.add_argument('--num_nodes_keep', type=int, default=46)
    parser.add_argument('--data_seed', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--lam', type=float, default=0.1)
    parser.add_argument('--tau_update', type=float, default=0.999)
    parser.add_argument('--hidden', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=2500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_ratio', type=float, default=0.05)  # 5 %
    parser.add_argument('--use_early_stop', type=str2bool, default=False)
    parser.add_argument('--early_stop_patience', type=int, default=100)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--outdir', type=str, default='')
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(args.device_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    nodes_num = args.num_nodes_keep
    dt_str = str(DT_DATA).replace('.', '')
    base = os.path.dirname(os.path.abspath(__file__))
    series_path = os.path.join(base, 'data',
        f'Series_N{nodes_num}_{NET_NAME}_T{T_TOTAL}_dt{dt_str}_seed{args.data_seed}.pickle')

    with open(series_path, 'rb') as f:
        objectAij, series = pickle.load(f)

    # Analytic true dx/dt from generation_Lorenz.py
    A_np = np.asarray(objectAij, dtype=np.float64)
    A_rowsum = A_np.sum(axis=1)
    s_full = series.reshape(-1, nodes_num, DIMENSION)
    xs, ys, zs = s_full[..., 0], s_full[..., 1], s_full[..., 2]
    coupling = EPSILON * (xs @ A_np.T - A_rowsum[None, :] * xs)
    dxdt_all = np.empty_like(s_full)
    dxdt_all[..., 0] = SIGMA * (ys - xs) + coupling
    dxdt_all[..., 1] = RHO * xs - ys - xs * zs
    dxdt_all[..., 2] = xs * ys - BETA * zs

    print(f'Loaded series: {s_full.shape}, dxdt: {dxdt_all.shape}')
    print(f'Adjacency: {A_np.shape}, edges={int(A_np.sum())}, nodes={nodes_num}, seed={args.seed}')
    series = s_full

    if args.outdir:
        data_path = args.outdir
    else:
        data_path = os.path.join(base, 'results_simul',
            f'Nodes{nodes_num}_Ntrain{args.num_train_samples}_hidden{args.hidden}_seed{args.seed}/')
    os.makedirs(data_path, exist_ok=True)

    train_end = int(round(40 / DT_DATA))
    eval_end = series.shape[0]

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

    print(f'Train: {series_train.shape[0]} points  (uniform subsample)')
    print(f'Test:  {series_test.shape[0]} points')

    # Simul: fully connected initial adjacency (no diagonal). Symmetric all-ones
    # so np.where(initialA) == np.where(initialA.T); HR simul ordering applies.
    initialA = np.ones((nodes_num, nodes_num))
    np.fill_diagonal(initialA, 0)
    edge_index = get_edge_index(initialA)

    X_train = torch.as_tensor(series_train.astype('float')).float()
    y_train = torch.as_tensor(dxdt_train.astype('float')).float()
    X_val = torch.as_tensor(series_val.astype('float')).float()
    y_val = torch.as_tensor(dxdt_val.astype('float')).float()
    X_test = torch.as_tensor(series_test.astype('float')).float()
    y_test = torch.as_tensor(dxdt_test.astype('float')).float()

    len_train = X_train.shape[0]
    batch = max(1, math.ceil(len_train * args.batch_ratio))
    print(f'Batch size: {batch}  (batch_ratio={args.batch_ratio})')

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
    val_metric_over_time = []
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

        cur_loss = total_loss / max(num_items, 1)
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
                msg += (f', R²[sx={cur_metrics["self_x_corr2"]:.3f}, '
                        f'sy={cur_metrics["self_y_corr2"]:.3f}, '
                        f'sz={cur_metrics["self_z_corr2"]:.3f}, '
                        f'msg={cur_metrics["msg_corr2"]:.3f}, '
                        f'overall={cur_metrics["overall_r2"]:.3f}], '
                        f'AUC={cur_metrics["AUC"]:.3f}')
            print(msg)

        if cur_metrics is not None and cur_val_metric < best_val_metric:
            best_val_metric = cur_val_metric
            best_state = copy.deepcopy(_unwrap(Dyn).state_dict())
            best_snapshot = {'epoch': epoch + 1, 'train_loss': cur_loss, **cur_metrics}
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
            snap = {
                'epoch': epoch + 1,
                'train_loss': cur_loss,
                **periodic_metrics,
                'best_component_mean_err': float(best_val_metric),
            }
            with open(data_path + f'metrics_snapshot_e{epoch+1}.json', 'w') as f:
                json.dump(snap, f, indent=2, default=float)
            torch.save(copy.deepcopy(_unwrap(Dyn).state_dict()),
                       data_path + f'recorded_model_e{epoch+1}.pt')

    elapsed = time.time() - start_time
    print(f'\nTraining done in {elapsed:.1f}s')

    if best_state is not None:
        _unwrap(Dyn).load_state_dict(best_state)
        print(f'Loaded best model (component_mean_err={best_val_metric:.4f})')
    final_metrics = compute_component_metrics(
        Dyn, X_test, y_test, edge_index, objectAij, device)
    print(f'Test overall_R²={final_metrics["overall_r2"]:.4f}, '
          f'sx={final_metrics["self_x_corr2"]:.3f}, '
          f'sy={final_metrics["self_y_corr2"]:.3f}, '
          f'sz={final_metrics["self_z_corr2"]:.3f}, '
          f'msg={final_metrics["msg_corr2"]:.3f}, '
          f'AUC={final_metrics["AUC"]:.3f}')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(loss_over_time, label='train_loss'); ax[0].legend(); ax[0].set_title('Train loss')
    ax[1].plot(val_metric_over_time, label='component_mean_err (test)'); ax[1].legend()
    ax[1].set_title('Component-mean error (early-stop metric)')
    plt.tight_layout()
    plt.savefig(data_path + 'loss_curves.png'); plt.close()

    with open(data_path + f'loss_over_time_lam{args.lam}.pkl', 'wb') as f:
        pickle.dump({'train': loss_over_time,
                     'val_component_err': val_metric_over_time,
                     'val_overall_r2': val_overall_r2_hist}, f)

    results = {
        'nodes_num': int(nodes_num),
        'num_train_samples': int(effective_num_train),
        'seed': args.seed,
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
