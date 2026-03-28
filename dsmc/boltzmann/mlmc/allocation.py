"""
Giles' optimal sample-count allocation for Multilevel Monte Carlo.
"""
import math


def giles_optimal_allocation(
    variances: list,
    costs: list,
    epsilon: float,
) -> list:
    """Compute optimal per-level sample counts (Giles 2008).

    Minimises total cost  C = Σ n_ℓ C_ℓ  subject to estimator variance
    Σ V_ℓ / n_ℓ  ≤  ε² / 2  (the other half of ε² is reserved for bias).

    The optimum is:

        n_ℓ = (2/ε²) · √(V_ℓ/C_ℓ) · Σ_k √(V_k C_k)

    Parameters
    ----------
    variances : list of float, length L+1
        Estimated variance of the level-ℓ correction:
        V_0 = Var(Q_0),  V_ℓ = Var(Q_ℓ − Q_{ℓ−1}) for ℓ ≥ 1.
    costs : list of float, length L+1
        Computational cost per sample at each level.
    epsilon : float
        Target root-mean-squared error.

    Returns
    -------
    list of int
        n_samples[ℓ] = optimal number of samples at level ℓ (≥ 2).
    """
    if len(variances) != len(costs):
        raise ValueError("variances and costs must have the same length")
    mu = sum(math.sqrt(v * c) for v, c in zip(variances, costs))
    scale = 2.0 / (epsilon ** 2)
    n_raw = [scale * math.sqrt(v / c) * mu for v, c in zip(variances, costs)]
    return [max(2, int(math.ceil(n))) for n in n_raw]


def level_cost(nlocal_base: int, level: int, nsteps: int, size: int = 1) -> float:
    """Model cost for one MLMC sample at *level*.

    Level 0 is a standalone simulation; level ℓ ≥ 1 is a coupled
    (fine, coarse) pair so its cost includes both:

        level 0 :  N_0 · nsteps
        level ℓ :  (N_ℓ + N_{ℓ−1}) · nsteps  =  N_0 · 2^ℓ · 1.5 · nsteps

    Parameters
    ----------
    nlocal_base : int
        Particles per MPI rank at level 0.
    level : int
        Level index (0-based).
    nsteps : int
        Number of time steps per simulation.
    size : int
        Number of MPI ranks (multiplied in to get total particles).
    """
    N_fine = nlocal_base * (2 ** level) * size
    if level == 0:
        return float(N_fine * nsteps)
    N_coarse = N_fine // 2
    return float((N_fine + N_coarse) * nsteps)
