"""
Training Script: Rulkov chemical-synapse simultaneous inference
===============================================================
Factorized-message variant:

    message(x_i, x_j) = msg_recv(x_i) * msg_send(x_j)

Two independent MLPs replace the joint msg_fnc([x_i, x_j]), matching the
receiver/sender product form of the true chemical synapse message. Component
targets are unchanged from the _new script:

    self_x(x, y) = (alpha / (1 + x^2) + y - x) / dt
    self_y(x, y) = -(mu / dt) * (x - sigma)
    message(x_i, x_j) = -(gc / dt) * (x_i - v_s) * Gamma(x_j)

The aggregated message is added only to the x dimension. Default output
directory is results_simul_factmsg/ (distinct from results_simul_new/).
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
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import (
    tau_soft_regu_deriv_MP_factored,
    GraphDataset,
    get_edge_index,
    calculate_auc,
    count_parameters,
    uniform_subsample_indices,
    evaluate_on_set,
    safe_corrcoef,
)


NET_NAME = 'RulkovChemicalODE'
T_TOTAL = 100
DT_DATA = 0.001  # default for the discrete-dyn pipeline; overridden by --dt_data
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
def compute_component_metrics(Dyn, X_set, y_set, edge_index, objectAij, device, gc, v_s, dyn_lam, theta):
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
    xi_all = X_dev[:, tgt_idx, 0]
    xj_all = X_dev[:, src_idx, 0]
    xi_flat = xi_all.reshape(-1)
    xj_flat = xj_all.reshape(-1)
    G_pred_flat = (core.msg_recv(xi_flat.unsqueeze(-1)) *
                   core.msg_send(xj_flat.unsqueeze(-1))).reshape(-1)

    soft_w = F.softmax(core.weights / core.tau, dim=1)[:, 0]
    soft_w_tiled = soft_w.unsqueeze(0).expand(T, -1).reshape(-1)
    msg_pred_weighted = (soft_w_tiled * G_pred_flat).cpu().numpy()

    objA = np.asarray(objectAij)
    mask = np.ones_like(objA, dtype=bool)
    np.fill_diagonal(mask, 0)
    aij_true = objA.T[mask].astype(np.float32)
    xi_n = xi_flat.cpu().numpy()
    xj_n = xj_flat.cpu().numpy()
    z = np.clip(dyn_lam * (xj_n - theta), -60.0, 60.0)
    gamma_xj = 1.0 / (1.0 + np.exp(-z))
    G_true_flat = -(gc / DT_DATA) * (xi_n - v_s) * gamma_xj
    aij_tiled = np.tile(aij_true, T)
    msg_true_weighted = aij_tiled * G_true_flat

    sx_c2 = safe_corrcoef(sx_true, sx) ** 2
    sy_c2 = safe_corrcoef(sy_true, sy) ** 2
    msg_c2 = safe_corrcoef(msg_true_weighted, msg_pred_weighted) ** 2

    c2s = [sx_c2, sy_c2, msg_c2]
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
        'msg_corr2': float(msg_c2),
        'component_mean_err': comp_err,
        'AUC': float(auc_v),
        'AUPRC': float(auprc_v),
    }


def main():
    global DT_DATA
    parser = argparse.ArgumentParser(description="Rulkov chemical simultaneous inference")
    parser.add_argument('--num_train_samples', type=int, default=2000)
    parser.add_argument('--num_nodes_keep', type=int, default=100)
    parser.add_argument('--data_seed', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--lam', type=float, default=0.01)
    parser.add_argument('--tau_update', type=float, default=0.999)
    parser.add_argument('--v_s', type=float, default=V_S)
    parser.add_argument('--dyn_lam', type=float, default=LAM)
    parser.add_argument('--theta', type=float, default=THETA)
    parser.add_argument('--hidden', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=2500)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch_ratio', type=float, default=0.0)
    parser.add_argument('--square', type=str2bool, default=True,
                        help='If True, use squared (MSE) loss; if False, use L1 (MAE) loss')
    parser.add_argument('--legacy_loss', type=str2bool, default=False,
                        help='If True, use legacy sum-style loss (sum(err) + lam*sum(w)); else mean-relative loss')
    parser.add_argument('--train_t_start', type=float, default=20.0,
                        help='Train window start time (in same units as T_TOTAL)')
    parser.add_argument('--train_t_end', type=float, default=60.0,
                        help='Train window end time (exclusive)')
    parser.add_argument('--eval_t_start', type=float, default=60.0,
                        help='Eval (val+test) pool start time')
    parser.add_argument('--eval_t_end', type=float, default=90.0,
                        help='Eval (val+test) pool end time (exclusive)')
    parser.add_argument('--use_early_stop', type=str2bool, default=False)
    parser.add_argument('--early_stop_patience', type=int, default=100)
    parser.add_argument('--device_id', type=int, default=3)
    parser.add_argument('--outdir', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--file_tag', type=str, default='')
    parser.add_argument('--dt_data', type=float, default=0.001,
                        help='Physical time step of the stored series and TrueDxdt (default 0.001 '
                             'matches generation_RulkovChemical.py default). Used for index '
                             'arithmetic (t_start/dt_data) and for scaling the analytic '
                             '(sx_true, sy_true, G_true) against stored dxdt.')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader worker count. Set to 0 to disable multiprocessing '
                             '(slower but immune to fork+CUDA deadlocks seen in long-running '
                             'parallel sweeps).')
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
    # Auto-route: N<80 -> data_small, else data_large (unless --data_dir overrides).
    data_dir = args.data_dir or os.path.join(base, 'data_small' if nodes_num < 80 else 'data_large')
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
            base, 'results_simul',
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

    print(f'Train: {series_train.shape[0]} points  (uniform subsample)')
    print(f'Test:  {series_test.shape[0]} points')

    initialA = np.ones((nodes_num, nodes_num))
    np.fill_diagonal(initialA, 0)
    edge_index = get_edge_index(initialA)

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
    print(f'Batch size: {batch}  ({batch_note})')

    train_dataset = GraphDataset(X_train, y_train, edge_index)
    nw = args.num_workers
    trainloader = DataLoader(
        train_dataset, batch_size=batch, shuffle=False,
        num_workers=nw, pin_memory=True, persistent_workers=(nw > 0)
    )

    edge_num = edge_index.shape[1]
    tau = 1.0

    Dyn = tau_soft_regu_deriv_MP_factored(
        n_f=DIMENSION, msg_dim=1, ndim=DIMENSION, delt_t=DT_DATA,
        edge_num=edge_num, tau=tau, lam=args.lam,
        hidden=args.hidden, aggr='add'
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
                    loss = Dyn.loss(ginput, square=args.square, legacy=args.legacy_loss)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(Dyn.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                total_loss += loss.item()
                i += 1
                num_items += 1

        tau *= args.tau_update
        _unwrap(Dyn).update_tau(tau)

        cur_loss = total_loss / max(num_items, 1)
        loss_over_time.append(cur_loss)

        cur_val_metric = val_metric_over_time[-1] if val_metric_over_time else float('inf')
        cur_metrics = None
        if (epoch + 1) % VAL_EVERY == 0 or epoch == 0:
            cur_metrics = compute_component_metrics(
                Dyn, X_test, y_test, edge_index, objectAij, device,
                gc=actual_gc, v_s=args.v_s, dyn_lam=args.dyn_lam, theta=args.theta)
            cur_val_metric = cur_metrics['component_mean_err']
        val_metric_over_time.append(cur_val_metric)
        val_overall_r2_hist.append(cur_metrics['overall_r2'] if cur_metrics else
                                   (val_overall_r2_hist[-1] if val_overall_r2_hist else float('nan')))

        if (epoch + 1) % 100 == 0:
            msg = f'Epoch {epoch+1}: train_loss={cur_loss:.4f}, comp_err={cur_val_metric:.4f}'
            if cur_metrics is not None:
                msg += (f', corr²[sx={cur_metrics["self_x_corr2"]:.3f}, '
                        f'sy={cur_metrics["self_y_corr2"]:.3f}, '
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
                Dyn, X_test, y_test, edge_index, objectAij, device,
                gc=actual_gc, v_s=args.v_s, dyn_lam=args.dyn_lam, theta=args.theta)
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
        Dyn, X_test, y_test, edge_index, objectAij, device,
        gc=actual_gc, v_s=args.v_s, dyn_lam=args.dyn_lam, theta=args.theta)
    print(f'Test overall_R²={final_metrics["overall_r2"]:.4f}, '
          f'sx={final_metrics["self_x_corr2"]:.3f}, '
          f'sy={final_metrics["self_y_corr2"]:.3f}, '
          f'msg={final_metrics["msg_corr2"]:.3f}, '
          f'AUC={final_metrics["AUC"]:.3f}')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(loss_over_time, label='train_loss')
    ax[0].legend()
    ax[0].set_title('Train loss')
    ax[1].plot(val_metric_over_time, label='component_mean_err (test)')
    ax[1].legend()
    ax[1].set_title('Component-mean error (early-stop metric)')
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
        'v_s': args.v_s,
        'dyn_lam': args.dyn_lam,
        'theta': args.theta,
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
