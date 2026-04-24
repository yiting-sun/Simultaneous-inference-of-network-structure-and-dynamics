"""
Training Script: Rulkov chemical-synapse known-structure dynamics inference
===========================================================================
Factorized-message variant:

    message(x_i, x_j) = msg_recv(x_i) * msg_send(x_j)

Two independent MLPs replace the joint msg_fnc([x_i, x_j]), matching the
receiver/sender product form of the true chemical synapse message. Component
targets are unchanged from the _new script:

    self_x(x, y) = (alpha / (1 + x^2) + y - x) / dt
    self_y(x, y) = -(mu / dt) * (x - sigma)
    message(x_i, x_j) = -(gc / dt) * (x_i - v_s) * Gamma(x_j)

Default output directory is results_knstruc_factmsg/ (distinct from the
original _new script's results_knstruc_new/, so both runs coexist).

Data split:
    Train: time indices in T in [30, 50)
    Val/Test: from time indices in T in [50, 70), sample 500 unique times for
              val and another 500 unique times for test (1000 unique total).
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
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import (
    Graph_deriv_NN_factored,
    GraphDataset,
    get_edge_index,
    count_parameters,
    uniform_subsample_indices,
    evaluate_on_set,
    safe_corrcoef,
)


NET_NAME = 'RulkovChemicalODE'
T_TOTAL = 100
DT_DATA = 0.01
DIMENSION = 2

ALPHA = 4.3
SIGMA = -1.6
MU = 0.001
V_S = 2.0
LAM = 10.0
THETA = -1.0


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


def float_to_tag(value):
    return format(value, "g").replace(".", "")


def gamma_fn(x):
    z = np.clip(LAM * (x - THETA), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


@torch.no_grad()
def compute_component_metrics(Dyn, X_set, y_set, edge_index, device, gc):
    core = _unwrap(Dyn)
    core.eval()

    overall = evaluate_on_set(Dyn, X_set, y_set, edge_index, device)

    T = X_set.shape[0]
    N = X_set.shape[1]
    ei = edge_index.to(device)
    src_idx = ei[0]
    tgt_idx = ei[1]

    x_flat = X_set.to(device).reshape(T * N, DIMENSION)
    sx = core.node_fnc_x(x_flat).reshape(-1).cpu().numpy()
    sy = core.node_fnc_y(x_flat).reshape(-1).cpu().numpy()
    xa = x_flat.cpu().numpy()
    x = xa[:, 0]
    y = xa[:, 1]

    sx_true = (ALPHA / (1.0 + x * x) + y - x) / DT_DATA
    sy_true = -(MU / DT_DATA) * (x - SIGMA)

    X_dev = X_set.to(device)
    xi_all = X_dev[:, tgt_idx, 0].reshape(-1)
    xj_all = X_dev[:, src_idx, 0].reshape(-1)
    G_pred = (core.msg_recv(xi_all.unsqueeze(-1)) *
              core.msg_send(xj_all.unsqueeze(-1))).reshape(-1).cpu().numpy()

    xi_n = xi_all.cpu().numpy()
    xj_n = xj_all.cpu().numpy()
    G_true = -(gc / DT_DATA) * (xi_n - V_S) * gamma_fn(xj_n)

    sx_c2 = safe_corrcoef(sx_true, sx) ** 2
    sy_c2 = safe_corrcoef(sy_true, sy) ** 2
    msg_c2 = safe_corrcoef(G_true, G_pred) ** 2

    c2s = [sx_c2, sy_c2, msg_c2]
    comp_err = float(np.nanmean([1.0 - c for c in c2s]))

    return {
        'overall_r2': overall['r2'],
        'node_r2_mean': overall['node_r2_mean'],
        'self_x_corr2': float(sx_c2),
        'self_y_corr2': float(sy_c2),
        'msg_corr2': float(msg_c2),
        'component_mean_err': comp_err,
    }


def main():
    global DT_DATA
    parser = argparse.ArgumentParser(description="Rulkov chemical known-structure inference")
    parser.add_argument('--num_train_samples', type=int, default=2000)
    parser.add_argument('--num_nodes_keep', type=int, default=50)
    parser.add_argument('--data_seed', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--hidden', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=2500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_ratio', type=float, default=0.05)
    parser.add_argument('--use_early_stop', type=str2bool, default=False)
    parser.add_argument('--early_stop_patience', type=int, default=100)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--outdir', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='',
                        help='Override data directory (defaults to data_small if N<80 else data_large).')
    parser.add_argument('--dt_data', type=float, default=0.001,
                        help='Physical time step of stored series/TrueDxdt (dt=0.001 in default pipeline).')
    parser.add_argument('--train_t_start', type=float, default=30.0)
    parser.add_argument('--train_t_end', type=float, default=50.0)
    parser.add_argument('--eval_t_start', type=float, default=50.0)
    parser.add_argument('--eval_t_end', type=float, default=70.0)
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers. Set 0 if deadlock seen in long parallel sweeps.')
    args = parser.parse_args()
    DT_DATA = args.dt_data

    if torch.cuda.is_available():
        torch.cuda.set_device(args.device_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    nodes_num = args.num_nodes_keep
    dt_str = float_to_tag(DT_DATA)
    base = os.path.dirname(os.path.abspath(__file__))
    # Auto-route: N<80 -> data_small, else data_large (unless --data_dir overrides)
    if args.data_dir:
        data_dir = args.data_dir
    else:
        data_dir = os.path.join(base, 'data_small' if nodes_num < 80 else 'data_large')
    series_pattern = os.path.join(
        data_dir, f'Series_N{nodes_num}_{NET_NAME}_T{T_TOTAL}_dt{dt_str}_gc*_d*_seed{args.data_seed}.pickle')
    series_matches = sorted(glob.glob(series_pattern))
    if len(series_matches) != 1:
        raise FileNotFoundError(
            f'Expected exactly one series file for pattern {series_pattern}, found {len(series_matches)}: {series_matches}')
    series_path = series_matches[0]
    match = re.search(r'_gc([^_]+)_d([^_]+)_seed', os.path.basename(series_path))
    if match is None:
        raise ValueError(f'Could not parse gc/degree tags from {series_path}')
    gc_tag = match.group(1)
    degree_tag = match.group(2)
    actual_gc = float(gc_tag.replace('p', '.'))
    dxdt_path = os.path.join(
        data_dir, f'TrueDxdt_N{nodes_num}_{NET_NAME}_T{T_TOTAL}_dt{dt_str}_gc{gc_tag}_d{degree_tag}_seed{args.data_seed}.pickle')

    with open(series_path, 'rb') as f:
        objectAij, series = pickle.load(f)
    with open(dxdt_path, 'rb') as f:
        _, dxdt_all = pickle.load(f)

    s_full = series.reshape(-1, nodes_num, DIMENSION)
    dxdt_all = dxdt_all.reshape(-1, nodes_num, DIMENSION)
    A_np = np.asarray(objectAij, dtype=np.float64)

    print(f'Loaded series: {s_full.shape}, dxdt: {dxdt_all.shape}')
    print(f'Adjacency: {A_np.shape}, edges={int(A_np.sum())}, nodes={nodes_num}, seed={args.seed}')

    if args.outdir:
        data_path = args.outdir
    else:
        data_path = os.path.join(
            base, 'results_knstruc',
            f'Nodes{nodes_num}_d{degree_tag}_gc{gc_tag}_Ntrain{args.num_train_samples}_hidden{args.hidden}_seed{args.seed}/')
    os.makedirs(data_path, exist_ok=True)

    train_start = int(round(args.train_t_start / DT_DATA))
    train_end = int(round(args.train_t_end / DT_DATA))
    eval_start = int(round(args.eval_t_start / DT_DATA))
    eval_end = int(round(args.eval_t_end / DT_DATA))
    series_train_full = s_full[train_start:train_end]
    dxdt_train_full = dxdt_all[train_start:train_end]

    eval_pool_size = eval_end - eval_start
    rng_valtest = np.random.RandomState(2024)
    eval_indices = rng_valtest.choice(eval_pool_size, size=1000, replace=False)
    val_indices = np.sort(eval_indices[:500]) + eval_start
    test_indices = np.sort(eval_indices[500:]) + eval_start

    series_val = s_full[val_indices]
    dxdt_val = dxdt_all[val_indices]
    series_test = s_full[test_indices]
    dxdt_test = dxdt_all[test_indices]

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

    edge_index = get_edge_index(A_np.T)

    X_train = torch.as_tensor(series_train.astype('float')).float()
    y_train = torch.as_tensor(dxdt_train.astype('float')).float()
    X_val = torch.as_tensor(series_val.astype('float')).float()
    y_val = torch.as_tensor(dxdt_val.astype('float')).float()
    X_test = torch.as_tensor(series_test.astype('float')).float()
    y_test = torch.as_tensor(dxdt_test.astype('float')).float()

    if args.batch_ratio == 0:
        batch = 128
        batch_note = 'fixed batch_size=128 because batch_ratio=0'
    else:
        batch = max(1, math.ceil(X_train.shape[0] * args.batch_ratio))
        batch_note = f'batch_ratio={args.batch_ratio}'
    init_lr = 5e-4 if batch < 10 else args.lr
    print(f'Batch size: {batch}  ({batch_note})')

    train_dataset = GraphDataset(X_train, y_train, edge_index)
    nw = args.num_workers
    trainloader = DataLoader(
        train_dataset, batch_size=batch, shuffle=False,
        num_workers=nw, pin_memory=True, persistent_workers=(nw > 0)
    )

    Dyn = Graph_deriv_NN_factored(
        n_f=DIMENSION, msg_dim=1, ndim=DIMENSION, delt_t=DT_DATA,
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
            cur_metrics = compute_component_metrics(Dyn, X_test, y_test, edge_index, device, actual_gc)
            cur_val_metric = cur_metrics['component_mean_err']
        val_metric_hist.append(cur_val_metric)
        val_overall_r2_hist.append(cur_metrics['overall_r2'] if cur_metrics else
                                   (val_overall_r2_hist[-1] if val_overall_r2_hist else float('nan')))

        if (epoch + 1) % 100 == 0:
            msg = f'Epoch {epoch+1}: train_loss={cur_loss:.4f}, comp_err={cur_val_metric:.4f}'
            if cur_metrics is not None:
                msg += (f', corr²[sx={cur_metrics["self_x_corr2"]:.3f}, '
                        f'sy={cur_metrics["self_y_corr2"]:.3f}, '
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
            periodic_metrics = compute_component_metrics(Dyn, X_test, y_test, edge_index, device, actual_gc)
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
    final_metrics = compute_component_metrics(Dyn, X_test, y_test, edge_index, device, actual_gc)
    print(f'Test overall_R²={final_metrics["overall_r2"]:.4f}, '
          f'corr²[sx={final_metrics["self_x_corr2"]:.3f}, '
          f'sy={final_metrics["self_y_corr2"]:.3f}, '
          f'msg={final_metrics["msg_corr2"]:.3f}]')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(train_loss_hist, label='train_loss')
    ax[0].legend()
    ax[0].set_title('Train loss')
    ax[1].plot(val_metric_hist, label='component_mean_err (test)')
    ax[1].legend()
    ax[1].set_title('Component-mean error (early-stop metric)')
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
