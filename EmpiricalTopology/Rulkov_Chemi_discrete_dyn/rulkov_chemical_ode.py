import os
from typing import Optional, Tuple

import numpy as np


NET_NAME = "RulkovChemicalODE"
DIMS = 2


def dt_to_str(dt: float) -> str:
    return str(dt).replace(".", "")


def chemical_activation(x: np.ndarray, lam: float, theta: float) -> np.ndarray:
    z = np.clip(lam * (x - theta), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def make_node_parameters(
    num_nodes: int,
    seed: int,
    alpha_base: float = 4.3,
    sigma_base: float = -1.6,
    mu_base: float = 0.001,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Deterministic node parameters.

    The formal default version uses the same alpha and sigma on every node.
    The previous heterogeneous version is intentionally kept below as comments
    in case it is needed later.
    """
    alpha = np.full(num_nodes, alpha_base, dtype=np.float64)
    sigma = np.full(num_nodes, sigma_base, dtype=np.float64)
    mu = np.full(num_nodes, mu_base, dtype=np.float64)
    return alpha, sigma, mu

    # Previous heterogeneous version, kept on purpose for possible reuse:
    # rng = np.random.default_rng(2024 + int(seed) * 97 + int(num_nodes))
    # alpha = alpha_base + rng.uniform(-0.08, 0.08, size=num_nodes)
    # sigma = sigma_base + rng.uniform(-0.03, 0.03, size=num_nodes)
    # mu = np.full(num_nodes, mu_base, dtype=np.float64)
    # return alpha.astype(np.float64), sigma.astype(np.float64), mu


def make_initial_state(num_nodes: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(4040 + int(seed) * 131 + int(num_nodes))
    init = np.zeros(num_nodes * DIMS, dtype=np.float64)
    init[0::2] = rng.uniform(-1.9, 1.5, size=num_nodes)
    init[1::2] = rng.uniform(-3.2, -1.8, size=num_nodes)
    return init


def reversal_potential(synapse_type: str, custom_vs: Optional[float]) -> float:
    if custom_vs is not None:
        return float(custom_vs)
    synapse_type = synapse_type.lower()
    if synapse_type == "excitatory":
        return 2.0
    if synapse_type == "inhibitory":
        return -1.8
    raise ValueError(f"Unsupported synapse_type={synapse_type!r}")


def select_kept_node_indices(
    total_nodes: int,
    num_keep: int,
    seed: int,
    full_adj: np.ndarray,
    max_retries: int = 10000,
) -> np.ndarray:
    if num_keep > total_nodes:
        raise ValueError(f"num_keep={num_keep} exceeds total_nodes={total_nodes}")
    if num_keep == total_nodes:
        return np.arange(total_nodes, dtype=np.int64)

    rng = np.random.default_rng(int(seed))
    for _ in range(max_retries):
        kept = np.sort(rng.choice(total_nodes, size=num_keep, replace=False)).astype(np.int64)
        if full_adj[np.ix_(kept, kept)].sum() > 0:
            return kept

    raise RuntimeError(
        f"Could not find a non-empty subgraph after {max_retries} retries "
        f"(num_keep={num_keep}, total_nodes={total_nodes}, seed={seed})."
    )


def load_adjacency(
    num_nodes: int,
    adj_path: Optional[str] = None,
    seed: int = 1,
) -> Tuple[np.ndarray, str, np.ndarray]:
    """
    A[i, j] means j -> i, matching the convention used elsewhere in the repo.

    If no adjacency is provided, use an all-zero matrix so the current default
    dataset stays desynchronized and topology-free.
    """
    if not adj_path:
        kept = np.arange(num_nodes, dtype=np.int64)
        return np.zeros((num_nodes, num_nodes), dtype=np.float64), "empty_zero_adjacency", kept

    ext = os.path.splitext(adj_path)[1].lower()
    if ext == ".npy":
        A = np.load(adj_path).astype(np.float64)
    elif ext == ".npz":
        with np.load(adj_path, allow_pickle=True) as data:
            if "A" in data:
                A = data["A"].astype(np.float64)
            else:
                first_key = list(data.keys())[0]
                A = data[first_key].astype(np.float64)
    else:
        raise ValueError(f"Unsupported adjacency format: {adj_path}")

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Adjacency must be square, got shape={A.shape}")

    total_nodes = A.shape[0]
    kept = select_kept_node_indices(total_nodes, num_nodes, seed, A)
    sub_A = A[np.ix_(kept, kept)]
    return sub_A, adj_path, kept


def rulkov_chemical_rhs(
    state: np.ndarray,
    A: np.ndarray,
    alpha: np.ndarray,
    sigma: np.ndarray,
    mu: np.ndarray,
    gc: float,
    v_s: float,
    lam: float,
    theta: float,
    ode_dt: float,
) -> np.ndarray:
    """
    Continuous-time embedding of the discrete chemical-synapse Rulkov map:

      x_{n+1,i} = alpha_i / (1 + x_{n,i}^2) + y_{n,i}
                  - gc * (x_{n,i} - v_s) * sum_j A[i,j] * Gamma(x_{n,j})
      y_{n+1,i} = y_{n,i} - mu_i * (x_{n,i} - sigma_i)

    with Gamma(x) = 1 / (1 + exp(-lambda * (x - theta))).

    We embed the map into an ODE by treating one map step as Euler integration
    with step ode_dt:

      dx_i/dt = (x_{n+1,i} - x_i) / ode_dt
      dy_i/dt = (y_{n+1,i} - y_i) / ode_dt

    This is an inference-based continuous approximation of the map, not the
    original formulation of the 2019 paper.
    """
    x = state[0::2]
    y = state[1::2]

    syn_input = A @ chemical_activation(x, lam=lam, theta=theta)
    x_next = alpha / (1.0 + x * x) + y - gc * (x - v_s) * syn_input
    y_next = y - mu * (x - sigma)

    rhs = np.empty_like(state)
    rhs[0::2] = (x_next - x) / ode_dt
    rhs[1::2] = (y_next - y) / ode_dt
    return rhs


def rulkov_chemical_rhs_batch(
    series: np.ndarray,
    A: np.ndarray,
    alpha: np.ndarray,
    sigma: np.ndarray,
    mu: np.ndarray,
    gc: float,
    v_s: float,
    lam: float,
    theta: float,
    ode_dt: float,
) -> np.ndarray:
    x = series[:, 0::2]
    y = series[:, 1::2]

    syn_input = chemical_activation(x, lam=lam, theta=theta) @ A.T
    x_next = alpha[None, :] / (1.0 + x * x) + y - gc * (x - v_s) * syn_input
    y_next = y - mu[None, :] * (x - sigma[None, :])

    rhs = np.empty_like(series)
    rhs[:, 0::2] = (x_next - x) / ode_dt
    rhs[:, 1::2] = (y_next - y) / ode_dt
    return rhs


def euler_rollout(
    init_state: np.ndarray,
    num_steps: int,
    dt: float,
    A: np.ndarray,
    alpha: np.ndarray,
    sigma: np.ndarray,
    mu: np.ndarray,
    gc: float,
    v_s: float,
    lam: float,
    theta: float,
) -> np.ndarray:
    series = np.empty((num_steps, init_state.size), dtype=np.float64)
    state = init_state.astype(np.float64).copy()

    for t in range(num_steps):
        series[t] = state
        state = state + dt * rulkov_chemical_rhs(
            state=state,
            A=A,
            alpha=alpha,
            sigma=sigma,
            mu=mu,
            gc=gc,
            v_s=v_s,
            lam=lam,
            theta=theta,
            ode_dt=dt,
        )
    return series


def euler_rollout_checked(
    init_state: np.ndarray,
    num_steps: int,
    dt: float,
    A: np.ndarray,
    alpha: np.ndarray,
    sigma: np.ndarray,
    mu: np.ndarray,
    gc: float,
    v_s: float,
    lam: float,
    theta: float,
    divergence_threshold: Optional[float] = None,
    ode_dt: Optional[float] = None,
) -> Tuple[np.ndarray, bool, Optional[int], float]:
    """
    Euler rollout with a lightweight stability guard.

    When `ode_dt` is None, behaves like before (`ode_dt == dt`, discrete-map
    iteration). When `ode_dt` is given (typically the physical map step 0.01),
    and `dt < ode_dt`, each Euler substep advances state by
    (dt/ode_dt)*(x_next - state), yielding a smoother continuous-time embedding
    that is numerically more stable for stiff regimes (e.g. high gc).
    """
    ode_dt_eff = dt if ode_dt is None else ode_dt
    series = np.empty((num_steps, init_state.size), dtype=np.float64)
    state = init_state.astype(np.float64).copy()
    max_abs_state = 0.0

    for t in range(num_steps):
        series[t] = state
        cur_abs = float(np.max(np.abs(state)))
        max_abs_state = max(max_abs_state, cur_abs)
        if not np.all(np.isfinite(state)):
            return series[: t + 1].copy(), False, t, max_abs_state
        if divergence_threshold is not None and cur_abs > divergence_threshold:
            return series[: t + 1].copy(), False, t, max_abs_state

        state = state + dt * rulkov_chemical_rhs(
            state=state,
            A=A,
            alpha=alpha,
            sigma=sigma,
            mu=mu,
            gc=gc,
            v_s=v_s,
            lam=lam,
            theta=theta,
            ode_dt=ode_dt_eff,
        )

    if not np.all(np.isfinite(state)):
        return series.copy(), False, num_steps, max_abs_state
    if divergence_threshold is not None:
        final_abs = float(np.max(np.abs(state)))
        max_abs_state = max(max_abs_state, final_abs)
        if final_abs > divergence_threshold:
            return series.copy(), False, num_steps, max_abs_state

    return series, True, None, max_abs_state
