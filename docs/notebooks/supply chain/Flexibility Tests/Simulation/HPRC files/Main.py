from ppopt.mplp_program import MPLP_Program
from ppopt.mpmodel import MPModeler
from ppopt.mp_solvers.solve_mpqp import solve_mpqp, mpqp_algorithm
from typing import List, Callable, Union
from collections import defaultdict
from numpy.polynomial.legendre import leggauss
from scipy.optimize import linprog
import itertools as itools
from sympy.logic.boolalg import BooleanTrue, BooleanFalse
import numpy as np
import chaospy as cp
import pickle
import time
import warnings
from concurrent.futures import as_completed, ThreadPoolExecutor
from functools import partial
import os
from scipy.stats import qmc

# Case Study
# Bansal (2000) Process Example 1
y_dict = {
    (0, 0, 0): 0.001,
    (0, 0, 1): 0.003,
    (0, 1, 0): 0.006,
    (1, 0, 0): 0.010,
    (0, 1, 1): 0.040,
    (1, 0, 1): 0.066,
    (1, 1, 0): 0.114,
    (1, 1, 1): 0.760
}

t_bounds = [(8, 16), (3, 11)]
d_bounds = [(0, 10), (0, 10), (0, 10)]
nt = len(t_bounds)
nd = len(d_bounds)

feasibility_algo = mpqp_algorithm.combinatorial_parallel
theta_bounds_algo = mpqp_algorithm.combinatorial_parallel

feas_algo_name = 'gp' if feasibility_algo.name in ['geometric_parallel', 'geometric'] else ('cb' if feasibility_algo in ['combinatorial', 'combinatorial_parallel'] else '')
tb_algo_name = 'gp' if theta_bounds_algo.name in ['geometric_parallel', 'geometric'] else ('cb' if feasibility_algo in ['combinatorial', 'combinatorial_parallel'] else '')

quad_method = 'smolyak'
smolyak_level = 12
def create_flexibility_model(tbounds: list, dbounds: list, y_list: tuple = None):
    j1 = 0.92
    j2 = 0.85
    j3 = 0.75

    m = MPModeler()

    u = m.add_var(name='u')
    F1 = m.add_var(name="F1")
    F2 = m.add_var(name="F2")
    F3 = m.add_var(name="F3")
    F4 = m.add_var(name="F4")
    F5 = m.add_var(name="F5")
    F6 = m.add_var(name="F6")
    F7 = m.add_var(name="F7")

    S = m.add_param(name='S')
    D = m.add_param(name='D')

    d1 = m.add_param(name='d1')
    d2 = m.add_param(name='d2')
    d3 = m.add_param(name='d3')

    m.add_constr(F4 - j1 * F2 == 0)
    m.add_constr(F1 - F2 - F3 == 0)
    m.add_constr(F5 - j2 * F4 == 0)
    m.add_constr(F6 - j3 * F3 == 0)
    m.add_constr(F7 - F5 - F6 == 0)
    m.add_constr(F1 - S <= u)
    m.add_constr(D - F7 <= u)
    m.add_constr(F2 - d1 * y_list[0] <= u)
    m.add_constr(F4 - d2 * y_list[1] <= u)
    m.add_constr(F3 - d3 * y_list[2] <= u)

    # for v in [F1, F2, F3, F4, F5, F6, F7]:
    #     m.add_constr(v >= 0)

    m.add_constr(tbounds[0][0] + 1e-6 <= S)
    m.add_constr(S <= tbounds[0][1])
    m.add_constr(tbounds[1][0] + 1e-6 <= D)
    m.add_constr(D <= tbounds[1][1])

    m.add_constr(dbounds[0][0] <= d1)
    m.add_constr(d1 <= dbounds[0][1])
    m.add_constr(dbounds[1][0] <= d2)
    m.add_constr(d2 <= dbounds[1][1])
    m.add_constr(dbounds[2][0] <= d3)
    m.add_constr(d3 <= dbounds[2][1])

    m.set_objective(u)

    return m

def joint_pdf(theta: list):
    Sval, Dval = theta
    eps = 1e-12
    x = max(Sval - 8.0, eps)
    return (1 / (1.2 * np.pi * x)) * np.exp(
        -1.39 * (np.log(x)) ** 2 - 0.5 * (Dval - 7.0) ** 2)

def cost_function(d):
    coeffs = np.array([2,3,5])
    return float(np.dot(coeffs, d))

# Helper functions
def mpformulate_theta_bounds(flex_sol, num_theta: int, theta_bounds: list, num_design: int = 0, design_bounds: list = None, psi_idx: int = 0, theta_m: int = 0):
    A0, b0, F0 = np.empty((len(flex_sol), num_theta)), np.empty(
        (len(flex_sol), 1)), np.empty((len(flex_sol), num_design))
    num_cr = len(flex_sol.critical_regions)
    for i, region in enumerate(flex_sol.critical_regions):
        A0[i] = region.A[psi_idx, :num_theta]
        b0[i] = -region.b[psi_idx]
        F0[i] = -region.A[psi_idx, num_theta:num_theta+num_design]
    # print(f'num_cr:{num_cr}')
    # print(f'num_theta:{num_theta}')
    # print(f'num_design:{num_design}')
    # print(f"A0: {A0}")
    # print(f"b0: {b0}")
    # print(f"F0: {F0}")

    c = np.hstack([np.array([-1, 1]).reshape(1, -1),
                  np.zeros((1, 2 * (num_theta - 1 - theta_m)))]).reshape(-1, 1)
    # print(f'c:{c}')
    # print(f'c.shape: {c.shape}')

    row1_block = np.hstack([block for i in range(theta_m, num_theta)
                           for block in (A0[:, [i]], np.zeros((num_cr, 1)))])
    row2_block = np.hstack([block for i in range(theta_m, num_theta)
                           for block in (np.zeros((num_cr, 1)), A0[:, [i]])])
    bound_row = np.hstack([np.array([-1, 1]).reshape(1, -1),
                          np.zeros((1, 2 * (num_theta - 1 - theta_m)))])
    A = np.vstack([row1_block, row2_block, bound_row, -
                  np.eye(2*(num_theta-theta_m)), np.eye(2*(num_theta-theta_m))])
    # print(f'A: {A}')
    # print(f'A.shape: {A.shape}')

    x_lb = np.array([val for i in range(theta_m, len(theta_bounds))
                    for val in [theta_bounds[i][0]] * 2])
    x_ub = np.array([val for i in range(theta_m, len(theta_bounds))
                    for val in [theta_bounds[i][1]] * 2])
    b = np.vstack([b0, b0, np.zeros((1, 1)), -
                  x_lb.reshape(-1, 1), x_ub.reshape(-1, 1)])
    # print(f'b: {b}')
    # print(f'b.shape: {b.shape}')

    if F0.size == 0 and theta_m == 0:
        # print('here')
        return A, b, c, np.array([]), np.array([]), np.array([]), np.array([])

    F = np.vstack([F0, F0, np.zeros((1, num_design)), np.zeros(
        (4*(num_theta-theta_m), num_design))]) if num_design > 0 else np.vstack([F0, F0])
    # print(f'F:{F}')
    # print(f'F.shape: {F.shape}')
    if theta_m > 0:
        F_lltheta = np.hstack([A0[:, [i]] for i in range(theta_m)])
        # print(f'F_lltheta: {F_lltheta}')
        # print(f'F_lltheta.shape: {F_lltheta.shape}')
        F = np.hstack([np.vstack([-F_lltheta, -F_lltheta, np.zeros((1, len(range(theta_m)))), np.zeros((4*(num_theta-theta_m), theta_m))]), F]
                      ) if F.size > 0 else np.vstack([-F_lltheta, -F_lltheta, np.zeros((1, len(range(theta_m)))), np.zeros((4*(num_theta-theta_m), theta_m))])
    # print(f'F:{F}')
    # print(f'F.shape: {F.shape}')

    H = np.zeros((2*(num_theta-theta_m), theta_m+num_design))
    # print(f'H:{H}')
    # print(f'H.shape: {H.shape}')

    A_t = np.vstack([-np.eye(theta_m+num_design), np.eye(theta_m+num_design)])
    # print(f'A_t:{A_t}')
    # print(f'A_t.shape: {A_t.shape}')

    theta_lb = np.array([-theta_bounds[i][0] for i in range(theta_m)] + ([-j[0] for j in design_bounds] if isinstance(design_bounds, list)
                                                                         else [])).reshape(-1, 1)
    theta_ub = np.array([theta_bounds[i][1] for i in range(theta_m)] + ([j[1] for j in design_bounds] if isinstance(design_bounds, list)
                                                                        else [])).reshape(-1, 1)

    b_t = np.vstack([theta_lb, theta_ub])
    # print(f'b_t:{b_t}')
    # print(f'b_t.shape: {b_t.shape}')

    return A, b, c, H, A_t, b_t, F

def get_theta_bounds(flex_sol, numt, tbounds, numd: int = 0, dbounds: list = None, mp_algo: mpqp_algorithm = mpqp_algorithm.combinatorial):
    theta_bound_dict = defaultdict(dict)
    prob_dict = defaultdict(dict)

    for i in range(numt):
        A, b, c, H, A_t, b_t, F = mpformulate_theta_bounds(
            flex_sol=flex_sol, num_theta=numt, num_design=numd, theta_bounds=tbounds, design_bounds=dbounds, theta_m=i)
        # print(f'A.shape:{A.shape}')
        # print(f'b.shape: {b.shape}')
        # print(f'F.shape: {F.shape}')
        if F.size != 0:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("error", category=UserWarning, message="The chebychev ball has either a radius of zero, or the problem is not feasible!")
                    prob = MPLP_Program(A=A, b=b, c=c, H=H, A_t=A_t, b_t=b_t, F=F)

                prob.process_constraints()
                solution = solve_mpqp(problem=prob, algorithm=mp_algo)

                prob_dict[f"t{i}"] = prob
                theta_bound_dict[f"t{i}"] = solution

            except UserWarning as w:
                # MPLP infeasible / degenerate
                print(f"[theta {i}] MPLP infeasible / zero Chebyshev ball: {w}")
                prob_dict[f"t{i}"] = None
                theta_bound_dict[f"t{i}"] = None

        else:
            # LP fallback branch via scipy.optimize.linprog
            linres = linprog(c=c, A_ub=A, b_ub=b)

            if not linres.success:
                print(f"[theta {i}] linprog failed: status={linres.status}, "f"message={linres.message}")
                prob_dict[f"t{i}"] = linres
                theta_bound_dict[f"t{i}"] = None
            else:
                prob_dict[f"t{i}"] = linres
                theta_bound_dict[f"t{i}"] = [linres.x[1], linres.x[0]]

        print(f"Finished solving for theta{i+1}")

    probs = [p for _, p in prob_dict.items()]
    sols = [sol for _, sol in theta_bound_dict.items()]
    return probs, sols


def get_bounds_regions(sols: List, min_idx: int = 1, max_idx: int = 0):
    t_bounds_list = []
    t_regions_list = []

    for theta_sol in sols:
        if theta_sol is None:
            t_bounds_list.append(None)
            t_regions_list.append(None)
            continue

        if not getattr(theta_sol, "critical_regions", None):
            t_bounds_list.append(None)
            t_regions_list.append(None)
            continue

        min_max_list = []
        region_list = []

        for cr in theta_sol.critical_regions:
            # Store bounds
            Ab = np.concatenate([cr.A, cr.b], axis=1)[:2]
            min_max_list.append([Ab[min_idx].tolist(), Ab[max_idx].tolist()])

            # Store region constraints
            Ef = np.concatenate([cr.E, -cr.f], axis=1)
            region_array = np.array([row.tolist() for row in Ef], dtype=float)
            region_list.append(region_array)

        # Append per-theta data
        t_bounds_list.append(np.array(min_max_list))
        # <-- each region is a 2D array
        t_regions_list.append(np.array(region_list, dtype=object))

    return t_bounds_list, t_regions_list


def generate_region_combos(region_sizes, n_gl):
    """Generate region index combinations based on critical region structure."""
    n_theta = len(region_sizes)
    region_combo_shape = []
    for k in range(n_theta):
        n_paths = int(np.prod(n_gl[:k])) if k > 0 else 1
        region_combo_shape.extend([range(region_sizes[k])] * n_paths)
    return list(itools.product(*region_combo_shape))

def affine_expr(coeffs, symbols):
    return sum(c * s for c, s in zip(coeffs[:-1], symbols)) + coeffs[-1]

def normalized_lhs(ineq):
    return ineq.lhs.expand() if hasattr(ineq, 'lhs') else None
def _prepare_state_data(state, tbounds, dbounds, *, solve_algo, theta_algo, log=False):
    """
    Build and solve the flexibility problem for one state, then return only valid theta-related data.

    Returns
    -------
    dict with keys:
        state
        flex_sol
        sol_list
        t_bounds_list
        t_regions_list
        filtered_solutions
        filtered_theta_bounds
        filtered_theta_regions
    or None if no valid theta regions exist.
    """
    state_model = create_flexibility_model(y_list=state, tbounds=tbounds, dbounds=dbounds)
    state_prob = state_model.formulate_problem()
    state_prob.process_constraints()

    flex_sol = solve_mpqp(problem=state_prob, algorithm=solve_algo)

    if log:
        print(
            f"Number of critical regions in for flexibility function for state {state}: {len(flex_sol.critical_regions)}")

    _, sol_list = get_theta_bounds(flex_sol=flex_sol, numt=nt, numd=nd, tbounds=t_bounds, dbounds=dbounds,
                                   mp_algo=theta_algo)

    t_bounds_list, t_regions_list = get_bounds_regions(sols=sol_list)

    filtered_solutions = []
    filtered_theta_bounds = []
    filtered_theta_regions = []

    for k, (sol, tb, tr) in enumerate(zip(sol_list, t_bounds_list, t_regions_list)):
        if sol is None or tb is None or tr is None:
            # mark as empty stage but preserve position
            filtered_solutions.append(None)
            filtered_theta_bounds.append(None)
            filtered_theta_regions.append(None)
        else:
            filtered_solutions.append(sol)
            filtered_theta_bounds.append(tb)
            filtered_theta_regions.append(tr)

    return {
        "state": state,
        "flex_sol": flex_sol,
        "sol_list": sol_list,
        "theta_bounds_list": t_bounds_list,
        "theta_regions_list": t_regions_list,
        "filtered_solutions": filtered_solutions,
        "filtered_theta_bounds": filtered_theta_bounds,
        "filtered_theta_regions": filtered_theta_regions,
    }


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

## Smolyak Quadrature
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

def calculate_sm_esf(y_d: dict, prepared_data_by_s: dict, d_v, s_level: int, ns_u=None, ws_u=None):
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
                joint_func=joint_pdf,
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

## Parallelizing

def generate_design_vectors_sobol(bounds, n_samples, scramble=True, seed=None):
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

def _sm_esf_worker(d_vector, y_d, prepared_data_by_s, s_level):
    esf_sm, sm_esf_by_state = calculate_sm_esf(
        y_d=y_d,
        prepared_data_by_s=prepared_data_by_s,
        d_v=d_vector,   # change if your actual argument name differs
        s_level=s_level,
    )

    return {
        "design_vector": np.array(d_vector),
        "cost": cost_function(d_vector),
        "esf_sm": esf_sm,
        # "sm_esf_by_state": sm_esf_by_state,
    }

def parallel_calculate_sm_esf_threads(
    design_vectors,
    y_d,
    load_prepared_data_by_s,
    s_level=16,
    max_workers=None,
    show_progress=True,
):
    """
    Parallel evaluation with preserved output order and optional tqdm progress bar.
    """
    if max_workers is None:
        max_workers = os.cpu_count() or 1

    worker = partial(
        _sm_esf_worker,
        y_d=y_d,
        prepared_data_by_s=load_prepared_data_by_s,
        s_level=s_level,
    )

    results = [None] * len(design_vectors)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(worker, d_vec): idx
            for idx, d_vec in enumerate(design_vectors)
        }

        iterator = as_completed(future_to_idx)

        # if show_progress:
        #     iterator = tqdm(
        #         iterator,
        #         total=len(design_vectors),
        #         desc="Evaluating design vectors"
        #     )

        for future in iterator:
            idx = future_to_idx[future]
            results[idx] = future.result()

    return results


prepared_data_by_state = {
    state: _prepare_state_data(
        state,
        tbounds=t_bounds,
        dbounds=d_bounds,
        solve_algo=feasibility_algo,
        theta_algo=theta_bounds_algo,
    )
    for state in y_dict
}

load_prepared_data_by_state = prepared_data_by_state

ns = 8
dvector_list_sobol = generate_design_vectors_sobol(d_bounds, n_samples=ns)

MC_sm_results = parallel_calculate_sm_esf_threads(
    design_vectors = dvector_list_sobol,
    y_d = y_dict,
    load_prepared_data_by_s = load_prepared_data_by_state,
    s_level=smolyak_level,
    max_workers=None,
    show_progress=True,
)

with open (f'test.pkl', 'wb') as f:
    pickle.dump(MC_sm_results, f)