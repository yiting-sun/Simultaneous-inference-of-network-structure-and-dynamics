"""
Training Script: Lorenz Known-Structure Dynamics Inference (component-R² validation)
=====================================================================================
Same skeleton as HR_dyn_new/train_knstruc_HR.py, but with Lorenz true components:

    node_fnc_x(x)    vs.   10*(y - x)
    node_fnc_y(x)    vs.   28*x - y - x*z
    node_fnc_z(x)    vs.   x*y - (8/3)*z
    msg_fnc([x_i, x_j]) vs.  0.2*(x_j - x_i)      # x_i = target, x_j = source

These match generation_Lorenz.py exactly and the reference's
`knstruc_batch.ipynb -> self_message_error` (σ=10, ρ=28, β=8/3, ε=0.2).

Data layout (Lorenz_dyn/data/Series_N{N}_Lorenz_T100_dt0001_seed{s}.pickle):
    T=100, dt=0.001  -> 100,000 time steps
    Train pool:  T=0..40   (first 40,000 steps; uniform subsample to num_train_samples)
    Val/Test:    T>40      (remaining 60,000; random 1000 split 500/500 per VALTEST_SEED)

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
import torch
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import (
    Graph_deriv_NN,
    GraphDataset,
    get_edge_index,
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

# Lorenz parameters (must match generation_Lorenz.py)
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
def compute_component_metrics(Dyn, X_set, y_set, edge_index, device):
    """Component R² for Lorenz: self_x/y/z + message."""
    core = _unwrap(Dyn)
    core.eval()

    overall = evaluate_on_set(Dyn, X_set, y_set, edge_index, device)

    T = X_set.shape[0]
    N = X_set.shape[1]
    ei = edge_index.to(device)
    src_idx = ei[0]    # source
    tgt_idx = ei[1]    # target

    # Self components (evaluated on every node-sample)
    x_flat = X_set.to(device).reshape(T * N, DIMENSION)
    sx = core.node_fnc_x(x_flat).reshape(-1).cpu().numpy()
    sy = core.node_fnc_y(x_flat).reshape(-1).cpu().numpy()
    sz = core.node_fnc_z(x_flat).reshape(-1).cpu().numpy()
    xa = x_flat.cpu().numpy()
    x, y, z = xa[:, 0], xa[:, 1], xa[:, 2]
    sx_true = SIGMA * (y - x)                 # 10*(y - x)
    sy_true = RHO * x - y - x * z             # 28*x - y - x*z
    sz_true = x * y - BETA * z                # x*y - (8/3)*z

    # Message component: inputs [x_i_dim0 (target), x_j_dim0 (source)]
    X_dev = X_set.to(device)
    xi_all = X_dev[:, tgt_idx, 0].reshape(-1)  # target
    xj_all = X_dev[:, src_idx, 0].reshape(-1)  # source
    tmp = torch.stack([xi_all, xj_all], dim=1)
    G_pred = core.msg_fnc(tmp).reshape(-1).cpu().numpy()
    xi_n = xi_all.cpu().numpy()
    xj_n = xj_all.cpu().numpy()
    G_true = EPSILON * (xj_n - xi_n)           # 0.2*(x_source - x_target)

    # Reference (knstruc_batch.ipynb::self_message_error) uses 1 - corr^2
    # rather than 1 - R^2 — forgives overall linear scale/offset so that the
    # metric captures "did the MLP learn the functional SHAPE?".
    sx_c2 = safe_corrcoef(sx_true, sx) ** 2
    sy_c2 = safe_corrcoef(sy_true, sy) ** 2
    sz_c2 = safe_corrcoef(sz_true, sz) ** 2
    msg_c2 = safe_corrcoef(G_true, G_pred) ** 2

    c2s = [sx_c2, sy_c2, sz_c2, msg_c2]
    comp_err = float(np.nanmean([1.0 - c for c in c2s]))

    return {
        'overall_r2': overall['r2'],
        'node_r2_mean': overall['node_r2_mean'],
        'self_x_corr2': float(sx_c2),
        'self_y_corr2': float(sy_c2),
        'self_z_corr2': float(sz_c2),
        'msg_corr2': float(msg_c2),
        'component_mean_err': comp_err,
    }


def main():
    parser = argparse.ArgumentParser(description="Lorenz known-structure inference")
    parser.add_argument('--num_train_samples', type=int, default=100)
    parser.add_argument('--num_nodes_keep', type=int, default=20)
    parser.add_argument('--data_seed', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)
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

    # --- Compute true dx/dt analytically (same form as generation_Lorenz.py) ---
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
        data_path = os.path.join(base, 'results_knstruc',
            f'Nodes{nodes_num}_Ntrain{args.num_train_samples}_hidden{args.hidden}_seed{args.seed}/')
    os.makedirs(data_path, exist_ok=True)

    # Train pool: T=0..40  ;  Val/Test pool: T>40
    train_end = int(round(40 / DT_DATA))      # 40000
    eval_end = series.shape[0]                # 100000 (whole remainder)

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

    print(f'Train: {series_train.shape[0]} points  (uniform subsample of {train_full_size})')
    print(f'Val:   {series_val.shape[0]} points')
    print(f'Test:  {series_test.shape[0]} points')

    # Edge index: our objectAij has row=dest, col=src (A[i,j]=1 means j->i).
    # Pass objectAij.T so that edge_index[0]=source, edge_index[1]=target.
    # This matches Lorenz_dyn/reference_codes/knstru_smalderiv.py convention:
    #     edge_index = np.where(objectAij.T)
    edge_index = get_edge_index(A_np.T)

    X_train = torch.as_tensor(series_train.astype('float')).float()
    y_train = torch.as_tensor(dxdt_train.astype('float')).float()
    X_val = torch.as_tensor(series_val.astype('float')).float()
    y_val = torch.as_tensor(dxdt_val.astype('float')).float()
    X_test = torch.as_tensor(series_test.astype('float')).float()
    y_test = torch.as_tensor(dxdt_test.astype('float')).float()

    len_train = X_train.shape[0]
    batch = max(1, math.ceil(len_train * args.batch_ratio))
    init_lr = 5e-4 if batch < 10 else args.lr
    print(f'Batch size: {batch}  (batch_ratio={args.batch_ratio})')

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
    val_metric_hist = []
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
                msg += (f', corr²[sx={cur_metrics["self_x_corr2"]:.3f}, '
                        f'sy={cur_metrics["self_y_corr2"]:.3f}, '
                        f'sz={cur_metrics["self_z_corr2"]:.3f}, '
                        f'msg={cur_metrics["msg_corr2"]:.3f}], '
                        f'overall_R²={cur_metrics["overall_r2"]:.3f}')
            print(msg)

        if cur_metrics is not None and cur_val_metric < best_val_metric:
            best_val_metric = cur_val_metric
            best_state = copy.deepcopy(_unwrap(Dyn).state_dict())
            best_snapshot = {
                'epoch': epoch + 1,
                'train_loss': cur_loss,
                'component_mean_err': float(cur_val_metric),
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
                Dyn, X_test, y_test, edge_index, device)
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
          f'corr²[sx={final_metrics["self_x_corr2"]:.3f}, '
          f'sy={final_metrics["self_y_corr2"]:.3f}, '
          f'sz={final_metrics["self_z_corr2"]:.3f}, '
          f'msg={final_metrics["msg_corr2"]:.3f}]')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(train_loss_hist, label='train_loss'); ax[0].legend(); ax[0].set_title('Train loss')
    ax[1].plot(val_metric_hist, label='component_mean_err (test)'); ax[1].legend()
    ax[1].set_title('Component-mean error (early-stop metric)')
    plt.tight_layout()
    plt.savefig(data_path + 'loss_curves.png'); plt.close()

    np.save(data_path + 'train_loss.npy', np.array(train_loss_hist))
    np.save(data_path + 'val_component_err.npy', np.array(val_metric_hist))
    np.save(data_path + 'val_overall_r2.npy', np.array(val_overall_r2_hist))

    results = {
        'nodes_num': int(nodes_num),
        'num_train_samples': int(effective_num_train),
        'seed': args.seed,
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
