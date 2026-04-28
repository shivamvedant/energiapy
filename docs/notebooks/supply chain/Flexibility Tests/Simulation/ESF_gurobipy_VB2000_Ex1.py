from typing import List, Callable
import numpy as np
import chaospy as cp
import pickle
import time
from concurrent.futures import as_completed, ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import os
from tqdm.auto import tqdm
from scipy.stats import qmc
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import r2_score, mean_squared_error
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt



#######################################################################################################################
# SMOLYAK QUADRATURE
#######################################################################################################################

def smolyak_nodes_weights(
    n_theta: int,
    level: int,
    rule: str = "gaussian",
    growth: bool = True,
):
    """
    Returns:
        ns_u: (N, n_theta) array of Smolyak nodes in u-space (each u_k in [-1,1])
        ws_u: (N,) array of weights for integrating over [-1,1]^n_theta
                   i.e., sum_i ws_u[i] * f(ns_u[i]) ≈ ∫_{[-1,1]^n} f(u) du
    """
    dist = cp.J(*[cp.Uniform(-1, 1) for _ in range(n_theta)])

    # Chaospy returns nodes shape (n_theta, N) and weights for expectation
    nodes, wE = cp.generate_quadrature(
        order=level,
        dist=dist,
        rule=rule,
        sparse=True,
        growth=growth,
    )

    nodes = np.asarray(nodes, dtype=float)        # (n_theta, N)
    wE = np.asarray(wE, dtype=float).ravel()      # (N,)

    ns_u = nodes.T                             # (N, n_theta)

    # Convert expectation weights to integral weights over [-1,1]^n:
    # E[f(U)] = ∫ f(u) p(u) du with p(u)=1/2^n on [-1,1]^n
    # => ∫ f(u) du = 2^n * E[f(U)]
    ws_u = (2.0 ** n_theta) * wE

    return ns_u, ws_u
def theta_interval_at_point(solution, theta_vector: np.ndarray, max_idx: int = 0, min_idx: int = 1) -> tuple:
    """Given the parametric solution for theta_k and the current 'state' vector (theta_prev + d),
        return the scalar lower and upper bound [t_min, t_max] for this theta_k.

    Args:
        solution (_type_): _description_
        t_vector (np.ndarray): _description_
        max_idx (int, optional): _description_. Defaults to 0.
        min_idx (int, optional): _description_. Defaults to 1.

    Returns:
        tuple: _description_
    """

    theta_vector_aug = np.append(theta_vector, 1).reshape(-1, 1)

    if isinstance(solution, list):
        theta_min = solution[0]
        theta_max = solution[1]
        return float(theta_min), float(theta_max)

    try:
        region = solution.get_region(theta_vector.reshape(-1, 1))
        coefficients = np.concatenate([region.A, region.b], axis=1)[:2, :]
        max_coefficients = coefficients[max_idx]
        min_coefficients = coefficients[min_idx]
        # theta_max = float(max_coefficients @ theta_vector_aug)
        # theta_min = float(min_coefficients @ theta_vector_aug)
        theta_max = (max_coefficients @ theta_vector_aug).item()
        theta_min = (min_coefficients @ theta_vector_aug).item()
        return theta_min, theta_max
    except:
        raise ValueError(
            "The provided theta_vector is not inside any critical region of the solution.")


def map_u_to_theta_and_jacobian(solutions, u, dvector):
    """
    Given stage-aligned parametric solutions for theta_k and the current
    disturbance vector d, return the theta vector and the Jacobian.

    Parameters
    ----------
    solutions : list
        Stage-aligned list. Each entry is either:
        - a valid parametric solution object for stage k, or
        - None for an empty / unusable stage.
    u : np.ndarray
        1D array of canonical coordinates.
    dvector : np.ndarray
        Current disturbance/design vector.

    Returns
    -------
    theta_values : np.ndarray
        Stage-aligned theta vector, same length as solutions.
    jacobian : float
    """
    theta_values = []
    jacobian = 1.0

    for k, sol in enumerate(solutions):
        if sol is None:
            theta_k = 0.0
            theta_values.append(theta_k)
            continue

        if isinstance(dvector, np.ndarray):
            thetavector = np.block([np.array(theta_values, dtype=float), dvector])
        else:
            thetavector = np.array(theta_values, dtype=float)

        thetamin, thetamax = theta_interval_at_point(sol, thetavector)
        length = thetamax - thetamin

        theta_k = 0.5 * length * u[k] + 0.5 * (thetamax + thetamin)
        theta_values.append(theta_k)

        jacobian *= 0.5 * length

    return np.array(theta_values, dtype=float), jacobian


def calculate_stocflexibility_smolyak(solutions: List, level: int, joint_func: Callable[[List[float]], float],
                                      d_vector: np.ndarray = None, rule: str = "gaussian", ns_u=None,
                                      ws_u=None) -> float:
    """Compute stochastic flexibility using a Smolyak sparse grid in canonical  u-space

    Args:
        solutions (List): list of solutions for each theta dimension (same structure as in calculate_stocflexibility)
        level (int): Smolyak level (1,2,3,...) controls accuracy & number of points
        joint_func (Callable[[List[float]], float]): callable f(theta_list) -> scalar
        d_vector (np.ndarray, optional):design vector (np.ndarray). Defaults to None.
        rule (str, optional): 1D quadrature rule passed to chaospy (e.g. "gaussian"). Defaults to "gaussian".

    Returns:
        float: _description_
    """

    n_theta = len(solutions)

    if ns_u is None or ws_u is None:
        ns_u, ws_u = smolyak_nodes_weights(
            n_theta=n_theta,
            level=level,
            rule=rule,
            growth=True,
        )

    start = time.time()
    stochastic_flexibility = 0.0

    for i in range(ns_u.shape[0]):
        u_vector = ns_u[i, :]
        theta_vector, jacobian = map_u_to_theta_and_jacobian(
            solutions, u_vector, d_vector)
        func_value = joint_func(theta_vector)
        stochastic_flexibility += func_value * jacobian * ws_u[i]

    end = time.time()
    print(
        f"Smolyak stochastic flexibility computed in {end - start:.4f} seconds.")
    return stochastic_flexibility

#######################################################################################################################
# EXPECTED STOCHASTIC FLEXIBILITY FUNCTION
#######################################################################################################################

def _safe_sf_call(func, state, label, **kwargs):
    """
    Safely evaluate a stochastic flexibility routine for one state.

    Returns
    -------
    float or None
        None means: skip this state's contribution.
    """
    try:
        return func(**kwargs)

    except ValueError as e:
        msg = str(e)

        known_skip_markers = (
            "not inside any critical region",
            "No region found that contains the given t_vector",
            "No solution available for this stage",
            "no valid theta regions",
            "no valid stages",
        )

        if any(marker in msg for marker in known_skip_markers):
            print(f"Skipping {label} SF for state {state}: {e}")
            return None

        raise

def calculate_sm_esf(y_d: dict, prepared_data_by_s: dict, d_v, s_level: int, joint_func, ns_u=None, ws_u=None):
    """
    Smolyak ESF using stage-aligned prepared data.
    Assumes map_u_to_theta_and_jacobian() now handles None entries in solutions.
    """
    sm_esf = 0.0
    esf_by_state = {}

    for s, p in y_d.items():
        try:
            data = prepared_data_by_s.get(s)

            if data is None:
                print(f"No valid theta regions for state {s}; skipping state")
                continue

            sols = data.get("filtered_solutions", None)

            if sols is None or len(sols) == 0:
                print(f"No filtered solutions for state {s}; skipping state")
                continue

            if not any(sol is not None for sol in sols):
                print(f"All filtered solutions are None for state {s}; skipping state")
                continue

            sf_idx_smolyak = _safe_sf_call(
                calculate_stocflexibility_smolyak,
                state=s,
                label="Smolyak",
                solutions=sols,
                level=s_level,
                joint_func=joint_func,
                d_vector=d_v,
                ns_u = ns_u,
                ws_u = ws_u,
            )

            if sf_idx_smolyak is not None:
                esf_by_state[s] = sf_idx_smolyak
                sm_esf += sf_idx_smolyak * p

        except ValueError as e:
            print(f"Skipping state {s} due to ValueError: {e}")
            continue

        print(f"Finished for state {s}.")

    return sm_esf, esf_by_state

#######################################################################################################################
# PARALLELIZATION
#######################################################################################################################

def generate_design_vectors_sobol(bounds, n_samples, scramble=True, seed=2):
    """
    Generate Sobol design vectors in the given bounds.

    Parameters
    ----------
    bounds : list[tuple[float, float]]
        [(lb1, ub1), (lb2, ub2), ...]
    n_samples : int
        Number of Sobol samples (total design vectors).
    scramble : bool, optional
        Whether to use scrambled Sobol (usually recommended).
    seed : int or None
        Random seed for reproducibility when scramble=True.

    Returns
    -------
    list[np.ndarray]
        List of design vectors as numpy arrays.
    """
    n_vars = len(bounds)

    # create Sobol engine in [0, 1]^n_vars
    engine = qmc.Sobol(d=n_vars, scramble=scramble, seed=seed)

    # draw n_samples points (uniform in [0,1]^d)
    u = engine.random(n=n_samples)  # shape: (n_samples, n_vars)

    # scale each dimension to its [lb, ub] interval
    lbs = np.array([b[0] for b in bounds])
    ubs = np.array([b[1] for b in bounds])

    samples = lbs + u * (ubs - lbs)

    return [samples[i, :] for i in range(n_samples)]

def _sm_esf_worker_mp(design_vector, y_d, prepared_data_by_s, s_level, cost_func, joint_func):
    """
    Top-level worker for multiprocessing.
    Must stay at module scope.
    """
    esf_sm, _ = calculate_sm_esf(
        y_d=y_d,
        prepared_data_by_s=prepared_data_by_s,
        d_v=design_vector,   # rename if your arg name differs
        s_level=s_level,
        joint_func=joint_func,
    )

    return {
        "design_vector": np.array(design_vector),
        "cost": cost_func(design_vector),
        "esf_sm": esf_sm,
    }

def parallel_calculate_sm_esf_processes(
        design_vectors,
        y_d,
        load_prepared_data_by_s,
        cost_func,
        joint_func,
        s_level=16,
        max_workers=None,
        show_progress=True,
):
    """
    Process-based parallel evaluation.
    Preserves result order to match design_vectors.
    """
    if max_workers is None:
        max_workers = os.cpu_count() or 1

    print(f'Using {max_workers} workers')

    worker = partial(
        _sm_esf_worker_mp,
        y_d=y_d,
        prepared_data_by_s=load_prepared_data_by_s,
        s_level=s_level,
        cost_func = cost_func,
        joint_func = joint_func
    )

    results = [None] * len(design_vectors)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(worker, d_vec): idx
            for idx, d_vec in enumerate(design_vectors)
        }

        iterator = as_completed(future_to_idx)

        if show_progress:
            iterator = tqdm(
                iterator,
                total=len(design_vectors),
                desc="Evaluating design vectors",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            )

        for future in iterator:
            idx = future_to_idx[future]
            results[idx] = future.result()

    return results