# src/optimizer.py
import numpy as np
from typing import Dict, List, Tuple, Optional
from numpy.linalg import norm
from dataclasses import dataclass

@dataclass
class OptimizationResult:
    """Container for optimization results."""
    scores: np.ndarray
    solutions: np.ndarray
    weights: np.ndarray

#def random_weights(n: int) -> np.ndarray:
 #   """Matches Julia's random_weights function."""
  #  w = np.random.rand(n)
   # return w/np.sum(w)

def weight_vectors_gaussian(npts: int, ndims: int) -> List[np.ndarray]: 
    pts = [np.random.randn(ndims) for _ in range(npts)]
    return [np.abs(v/norm(v)) for v in pts]

def weight_vectors_dirichlet(npts: int, ndims: int) -> List[np.ndarray]: 
    pts = [np.random.dirichlet(np.ones(ndims)) for _ in range(npts)]
    return [v for v in pts]

def weight_vectors_fibonacci(npts: int) -> List[np.ndarray]:
    total_pts = 8 * npts  
    indices = np.arange(0, total_pts, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / total_pts)
    theta = np.pi * (1 + 5**0.5) * indices
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    vectors = np.vstack((x, y, z)).T
    p_vectors = vectors[(vectors[:, 0] >= 0) & 
                            (vectors[:, 1] >= 0) & 
                            (vectors[:, 2] >= 0)]

    return [vec for vec in p_vectors[:npts]]

def optimize_with_weights(data: Dict[str, np.ndarray], 
                        objectives: List[str],
                        weights: np.ndarray,
                         sign_obj) -> np.ndarray:
    """Optimize objectives using given weights."""
    cleaned_data = {}
    for ob in objectives:
        if not isinstance(data[ob], np.ndarray):
            raise TypeError(f"Objective '{ob}' is not a NumPy array.")
        data[ob] = np.where(np.isfinite(data[ob]), data[ob], 0)

    weighted_sum = sum(w*data[ob]*s for w, ob,s in zip(weights, objectives, sign_obj))
    return np.argmax(weighted_sum, axis=1)

def optimize_with_transition(data: Dict[str, np.ndarray],
                             objectives: List[str],
                             weights: np.ndarray,
                             sign_obj: List[int],
                             T: float) -> np.ndarray:
    """
    Weighted argmax with a per-row feasibility mask: rows with transition_cost > T are invalid.
    data[...] shape must be (n_rows, n_alternatives).
    """
    for ob in objectives + ["transition_cost"]:
        if ob not in data:
            raise KeyError(f"Objective '{ob}' missing from data.")
        if not isinstance(data[ob], np.ndarray):
            raise TypeError(f"Objective '{ob}' is not a NumPy array.")
        data[ob] = np.where(np.isfinite(data[ob]), data[ob], 0)

    weighted_sum = sum(w * data[ob] * s for w, ob, s in zip(weights, objectives, sign_obj))
    mask = data["transition_cost"] > T 
    weighted_sum = weighted_sum.copy()
    weighted_sum[mask] = -np.inf

    all_bad = np.all(~np.isfinite(weighted_sum), axis=1)
    result = np.argmax(weighted_sum, axis=1)
    result[all_bad] = -1
    return result

def normalize_data_nadir(data: Dict[str, np.ndarray], 
                         objectives: List[str], 
                         sign_obj: List[int]) -> Tuple[Dict[str, np.ndarray], Dict[str, Tuple[float, float]]]:

    norm_data = {}
    nadir_bounds = {}

    for ob, s in zip(objectives, sign_obj):
        obj_vals = data[ob] * s
        zU = np.min(obj_vals)
        zN = np.max(obj_vals)
        norm_data[ob] = (obj_vals - zU) / (zN - zU + 1e-8)  
        nadir_bounds[ob] = (zU, zN)

    return norm_data

def normalize_data(data: Dict[str, np.ndarray], 
                  objectives: List[str]) -> Dict[str, np.ndarray]:
    """Normalize objective data."""
    norm_data = {}
    for ob in objectives:
        t = np.sum(np.max(data[ob], axis=1))  # Same as Julia's sum(maximum(data[ob], dims=2))
        norm_data[ob] = data[ob] / t if t != 0 else data[ob]
    return norm_data

def scores_for_sol(data: Dict[str, np.ndarray], 
                  objectives: List[str], 
                  sol: np.ndarray) -> np.ndarray:
    """Calculate scores for a given solution."""
    s = np.zeros(len(objectives))
    for i, ob in enumerate(objectives):
        s[i] = np.sum(data[ob][np.arange(len(sol)), sol])
    return s

def random_sample_frontier(data: Dict[str, np.ndarray],
                           objectives: List[str],
                           sign_obj: List[int],
                           npts: int,
                           *,
                           method: str = "gaussian",
                           T: float = np.inf,
                           include_endpoints: bool = False,
                           reportvars: List[str] | None = None,
                           **kwargs) -> Tuple[np.ndarray, np.ndarray]:

#def random_sample_frontier(data: Dict[str, np.ndarray],
 #                        objectives: List[str], sign_obj,
  #                       npts: int,
   #                      sense: str = 'maximize',
    #                     method: str = "gaussian",
     #                    T: float = np.inf, 
      #                   include_endpoints: bool = False,
       #                  **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate scores and solutions for points on the Pareto frontier.
    
    Optional kwargs:
        reportvars: List[str] - variables to report in scores matrix
        minimize: List[int] - indices of objectives to minimize
    """
    nobj = len(objectives)
    nparcels = data[objectives[0]].shape[0]
    norm_data = normalize_data(data, objectives)
    #normalize_data_nadir(data, objectives, sign_obj)
    
    if "gaussian" in method:
        weights = weight_vectors_gaussian(npts, nobj)
    elif "fibonacci_lattice" in method:
        weights = weight_vectors_fibonacci(npts)
    elif "dirichlet" in method:
        weights = weight_vectors_dirichlet(npts, nobj)
 
        
    if "transition_cost" not in data:
        raise KeyError("`data` must include a 'transition_cost' matrix of shape (n_rows, n_alternatives).")
    opt_data = dict(norm_data)
    opt_data["transition_cost"] = data["transition_cost"]

    if include_endpoints:
        for i in range(nobj):
            unit = np.zeros(nobj, dtype=float); unit[i] = 1.0
            weights.append(unit)

    reportvars = reportvars or objectives
    scores = np.zeros((len(weights), len(reportvars)), dtype=float)
    solutions = np.zeros((len(weights), nparcels), dtype=int)

    for i, w in enumerate(weights):
        sol = optimize_with_transition(opt_data, objectives, np.asarray(w), sign_obj, T)
        solutions[i] = sol
        scores[i] = scores_for_sol(data, reportvars, sol) 
    
    return OptimizationResult(scores, solutions, np.array(weights))


def get_baseline_scores(data: Dict[str, np.ndarray], 
                       objectives: List[str], 
                       baseline_index: int) -> np.ndarray:
    """Calculate baseline scores for each objective."""
    return np.array([np.sum(data[ob][:, baseline_index]) for ob in objectives])