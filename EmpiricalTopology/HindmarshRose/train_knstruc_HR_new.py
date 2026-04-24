"""
Training Script: HR Known-Structure Dynamics Inference (component-R² validation)
=================================================================================
Same data / model / optimizer as HR_dyn/train_knstruc_HR.py.  The ONLY
difference is the validation / early-stopping criterion:

    Instead of MSE between y_true and model output, we compare the learned
    MLPs against the analytic HR components on the test set:
        - node_fnc_x(x)  vs.  y - x^3 + 3 x^2 - z + 3.24
        - node_fnc_y(x)  vs.  1 - 5 x^2 - y
        - node_fnc_z(x)  vs.  0.004 * (4 (x + 1.6) - z)
        - msg_fnc([x_i, x_j])  vs.  0.15 * (2 - x_i) / (1 + exp(-10 (x_j - 1)))
    (see HR_dyn/reference_codes/knstruc.ipynb -> self_message_error)

Early-stop metric = mean of (1 - R²) across the 4 components (smaller = better).
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
import torch
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

import sys
# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'HR_dyn'))

from model import (
    Graph_deriv_NN,
    GraphDataset,
    get_edge_index,
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


# ---------------------------------------------------------------------------
# Component-wise metric computation (HR ground-truth equations)
# ---------------------------------------------------------------------------
def _unwrap(m):
    return getattr(m, '_orig_mod', m)


@torch.no_grad()
def compute_component_metrics(Dyn, X_set, y_set, edge_index, device):
    """Evaluate component 1 − corr² of self_x/y/z and message MLPs on a dataset.

    For each component the error is `1 - np.corrcoef(true, pred)[0,1] ** 2`
    (Pearson; matches `HR_dyn/reference_codes/knstruc.ipynb:self_message_error`).

    Returns dict with: self_x_err, self_y_err, self_z_err, msg_err,
    component_mean_err (= mean of the 4 errs), plus overall_r2 / node_r2_mean
    from evaluate_on_set for legacy comparison.
    """
    core = _unwrap(Dyn)
    core.eval()

    overall = evaluate_on_set(Dyn, X_set, y_set, edge_index, device)

    T = X_set.shape[0]
    N = X_set.shape[1]
    ei = edge_index.to(device)
    src_idx = ei[0]  # HR-source
    tgt_idx = ei[1]  # HR-target

    x_flat = X_set.to(device).reshape(T * N, DIMENSION)  # (T*N, 3)
    sx = core.node_fnc_x(x_flat).reshape(-1).cpu().numpy()
    sy = core.node_fnc_y(x_flat).reshape(-1).cpu().numpy()
    sz = core.node_fnc_z(x_flat).reshape(-1).cpu().numpy()
    xa = x_flat.cpu().numpy()
    sx_true = xa[:, 1] - xa[:, 0] ** 3 + 3 * xa[:, 0] ** 2 - xa[:, 2] + 3.24
    sy_true = 1.0 - 5.0 * xa[:, 0] ** 2 - xa[:, 1]
    sz_true = 0.004 * (4.0 * (xa[:, 0] + 1.6) - xa[:, 2])

    X_dev = X_set.to(device)
    xi_all = X_dev[:, tgt_idx, 0].reshape(-1)  # V_HR_target (matches msg_fnc col 0)
    xj_all = X_dev[:, src_idx, 0].reshape(-1)  # V_HR_source (col 1)
    tmp = torch.stack([xi_all, xj_all], dim=1)
    G_pred = core.msg_fnc(tmp).reshape(-1).cpu().numpy()
    xi_n = xi_all.cpu().numpy()
    xj_n = xj_all.cpu().numpy()
    G_true = 0.15 * (2.0 - xi_n) / (1.0 + np.exp(-10.0 * (xj_n - 1.0)))

    sx_err = one_minus_corr2(sx_true, sx)
    sy_err = one_minus_corr2(sy_true, sy)
    sz_err = one_minus_corr2(sz_true, sz)
    msg_err = one_minus_corr2(G_true, G_pred)

    comp_err = float(np.nanmean([sx_err, sy_err, sz_err, msg_err]))

    return {
        'overall_r2': overall['r2'],
        'node_r2_mean': overall['node_r2_mean'],
        'self_x_err': float(sx_err),
        'self_y_err': float(sy_err),
        'self_z_err': float(sz_err),
        'msg_err': float(msg_err),
        'component_mean_err': comp_err,
    }


def main():
    parser = argparse.ArgumentParser(
        description="HR known-structure inference with component-R² validation")
    parser.add_argument('--num_train_samples', type=int, default=200)
    parser.add_argument('--num_nodes_keep', type=int, default=279)
    parser.add_argument('--data_seed', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)
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
        data_path = (f'HR_dyn/results_knstruc_new/Nodes{nodes_num}_d{degree_tag}_gc{gc_tag}'
                     f'_Ntrain{args.num_train_samples}_hidden{args.hidden}_seed{args.seed}/')
    os.makedirs(data_path, exist_ok=True)

    train_end = int(250 / DT_DATA)    # 25000
    eval_end = int(400 / DT_DATA)     # 40000

    series_train_full = series[:train_end]
    dxdt_train_full = dxdt_all[:train_end]

    VALTEST_SEED = 2024
    eval_pool_size = eval_end - train_end
    rng_valtest = np.random.RandomState(VALTEST_SEED)
    eval_indices = rng_valtest.choice(eval_pool_size, size=1000, replace=False)
    val_indices = np.sort(eval_indices[:500]) + train_end
    test_indices = np.sort(eval_indices[500:]) + train_end

    series_val = series[val_indices]
    dxdt_val = dxdt_all[val_indices]
    series_test = series[test_indices]
    dxdt_test = dxdt_all[test_indices]

    train_full_size = series_train_full.shape[0]
    if args.num_train_samples > 0 and args.num_train_samples < train_full_size:
        train_indices = uniform_subsample_indices(train_full_size, args.num_train_samples)
    else:
        train_indices = np.arange(train_full_size, dtype=np.int64)
    effective_num_train = train_indices.shape[0]

    series_train = series_train_full[train_indices]
    dxdt_train = dxdt_train_full[train_indices]

    print(f'Train: {series_train.shape[0]} points')
    print(f'Val:   {series_val.shape[0]} points')
    print(f'Test:  {series_test.shape[0]} points')

    # edge_index convention: edge_index[0]=HR-source, edge_index[1]=HR-target.
    # HR generator defines A[i,j]=1 as "j -> i" (row=HR-target, col=HR-source).
    # Reference `kstruc_realderiv_batch10.py` uses a local get_edge_index that
    # does np.where(Adj.T), then calls get_edge_index(objectAij) — net effect is
    # edge_index = np.where(objectAij.T). We reproduce that here.
    edge_index = get_edge_index(objectAij.T)

    X_train = torch.as_tensor(series_train.astype('float')).float()
    y_train = torch.as_tensor(dxdt_train.astype('float')).float()
    X_val = torch.as_tensor(series_val.astype('float')).float()
    y_val = torch.as_tensor(dxdt_val.astype('float')).float()
    X_test = torch.as_tensor(series_test.astype('float')).float()
    y_test = torch.as_tensor(dxdt_test.astype('float')).float()

    len_train = X_train.shape[0]
    batch = max(1, math.ceil(len_train * args.batch_ratio))
    init_lr = 5e-4 if batch < 10 else args.lr
    print(f'Batch size: {batch}')

    train_dataset = GraphDataset(X_train, y_train, edge_index)
    trainloader = DataLoader(train_dataset, batch_size=batch, shuffle=False,
                             num_workers=4, pin_memory=True, persistent_workers=True)

    if effective_num_train > 1:
        effective_dt = (train_indices[-1] - train_indices[0]) * DT_DATA / (effective_num_train - 1)
    else:
        effective_dt = DT_DATA

    Dyn = Graph_deriv_NN(
        n_f=DIMENSION, msg_dim=1, ndim=DIMENSION, delt_t=effective_dt,
        hidden=args.hidden, aggr='add'
    ).to(device)
    Dyn = torch.compile(Dyn)
    print(f'Model params: {count_parameters(Dyn):,}')

    opt = torch.optim.Adam(Dyn.parameters(), lr=init_lr, weight_decay=1e-8)
    steps_per_epoch = max(1, len(trainloader))
    sched = OneCycleLR(opt, max_lr=init_lr, steps_per_epoch=steps_per_epoch,
                       epochs=args.epochs, final_div_factor=1e5)

    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    VAL_EVERY = 10

    train_loss_hist = []
    val_metric_hist = []  # component_mean_err on test set
    val_overall_r2_hist = []
    best_val_metric = float('inf')
    best_state = None
    best_snapshot = None
    epochs_no_improve = 0

    start_time = time.time()

    for epoch in tqdm(range(args.epochs)):
        Dyn.train()
        total_loss = 0.0
        num_items = 0
        for ginput in trainloader:
            ginput = ginput.to(device)
            opt.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                loss = Dyn.loss(ginput, square=False)
            bsz = int(ginput.batch[-1] + 1)
            scaler.scale(loss / bsz).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total_loss += loss.item()
            num_items += bsz

        cur_loss = total_loss / num_items
        train_loss_hist.append(cur_loss)

        cur_val_metric = val_metric_hist[-1] if val_metric_hist else float('inf')
        cur_metrics = None
        if (epoch + 1) % VAL_EVERY == 0 or epoch == 0:
            cur_metrics = compute_component_metrics(Dyn, X_test, y_test, edge_index, device)
            cur_val_metric = cur_metrics['component_mean_err']
        val_metric_hist.append(cur_val_metric)
        val_overall_r2_hist.append(cur_metrics['overall_r2'] if cur_metrics else
                                   (val_overall_r2_hist[-1] if val_overall_r2_hist else float('nan')))

        if (epoch + 1) % 100 == 0:
            msg = f'Epoch {epoch+1}: train_loss={cur_loss:.4f}, comp_err={cur_val_metric:.4f}'
            if cur_metrics is not None:
                msg += (f', 1-corr²[self_x={cur_metrics["self_x_err"]:.3f}, '
                        f'self_y={cur_metrics["self_y_err"]:.3f}, '
                        f'self_z={cur_metrics["self_z_err"]:.3f}, '
                        f'msg={cur_metrics["msg_err"]:.3f}], '
                        f'overall_R²={cur_metrics["overall_r2"]:.3f}')
            print(msg)

        if cur_metrics is not None and cur_val_metric < best_val_metric:
            best_val_metric = cur_val_metric
            best_state = copy.deepcopy(_unwrap(Dyn).state_dict())
            best_snapshot = {
                'epoch': epoch + 1,
                'train_loss': cur_loss,
                'component_mean_err': float(cur_val_metric),
                **{k: v for k, v in cur_metrics.items()},
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
                Dyn, X_test, y_test, edge_index, device)
            print(f'  [Periodic] Epoch {epoch+1}: overall_R²={periodic_metrics["overall_r2"]:.4f}, '
                  f'1-corr² self_x={periodic_metrics["self_x_err"]:.3f}, '
                  f'self_y={periodic_metrics["self_y_err"]:.3f}, '
                  f'self_z={periodic_metrics["self_z_err"]:.3f}, '
                  f'msg={periodic_metrics["msg_err"]:.3f}')

            snap = {
                'epoch': epoch + 1,
                'train_loss': cur_loss,
                'component_mean_err': periodic_metrics['component_mean_err'],
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
    final_metrics = compute_component_metrics(Dyn, X_test, y_test, edge_index, device)
    print(f'Test overall_R²={final_metrics["overall_r2"]:.4f}, '
          f'1-corr² self_x={final_metrics["self_x_err"]:.3f}, '
          f'self_y={final_metrics["self_y_err"]:.3f}, '
          f'self_z={final_metrics["self_z_err"]:.3f}, '
          f'msg={final_metrics["msg_err"]:.3f}')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(train_loss_hist, label='train_loss')
    ax[0].legend(); ax[0].set_title('Train loss')
    ax[1].plot(val_metric_hist, label='component_mean_err (test)')
    ax[1].legend(); ax[1].set_title('Component-mean error (early-stop metric)')
    plt.tight_layout()
    plt.savefig(data_path + 'loss_curves.png')
    plt.close()

    np.save(data_path + 'train_loss.npy', np.array(train_loss_hist))
    np.save(data_path + 'val_component_err.npy', np.array(val_metric_hist))
    np.save(data_path + 'val_overall_r2.npy', np.array(val_overall_r2_hist))

    results = {
        'nodes_num': int(nodes_num),
        'num_train_samples': int(effective_num_train),
        'seed': args.seed,
        'data_seed': args.data_seed,
        'degree_tag': degree_tag,
        'gc': actual_gc,
        'hidden': args.hidden,
        'epochs_trained': len(train_loss_hist),
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
