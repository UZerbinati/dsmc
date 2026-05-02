"""Discotic CFMZ DSMC solvers.

This module hosts solver classes for **discotic** liquid crystals — 2-D
disc-shaped (oblate-coin) particles whose orientation θ is the in-plane
angle of a labelled axis on the disc.  The disc is effectively 4-fold
symmetric (rotation by π/2 returns an equivalent configuration), so the
critical orientational order parameter is

    R_4 = |⟨e^{4iθ}⟩|

(rather than the rod's R_2), and the Onsager-type pair potential takes
the 4-fold form

    W_disc(θ₁, θ₂) = |sin(2(θ₁ − θ₂))|   .

The collision contact mechanics also change (see ``collision_disc.py``):
for hard discs the lever arms vanish, so collisions transfer no torque.

For now only the homogeneous (orientation-only) class lives here;
the inhomogeneous twin will be appended on the inhomogeneous-feature
branch.
"""
import os
import numpy as np
from mpi4py import MPI

from . import CFMZNeedleDSMCHomo, CFMZNeedleDSMC


def _disc_onsager_factory(L, bins, comm, n_fold=2):
    """Build ``(vlasov_force, interaction_energy)`` closures for the
    *homogeneous* disc Onsager pair potential
    ``W = |sin(m Δθ)|`` with ``m = n_fold / 2``.

    The default ``n_fold = 2`` corresponds to the calamitic-form
    Onsager kernel ``|sin(Δθ)|`` — the physically correct kernel
    for a 3-D discotic LC simulated in a 2-D domain, where θ is
    the projected disc normal (head-tail symmetric, so ``θ ≡ θ + π``).
    The unstable mode is ``cos(2θ)`` and the I-N transition produces
    the discotic nematic (N_D) phase with R₂ as the critical order
    parameter.

    Setting ``n_fold = 4`` recovers the 4-fold-symmetric kernel
    ``|sin(2 Δθ)|`` of the previous "2-D coin / square" interpretation,
    which gives a tetratic (R₄-driven) transition.

    Parameters
    ----------
    L : float
        Length-scale prefactor multiplying the Vlasov torque (kept
        named ``L`` for parity with the rod case; for discs this is
        the radius / characteristic disc size).
    bins : int
        Number of θ-grid cells used by the CIC density estimator and
        the convolution against W.
    comm : MPI.Comm
        Communicator over which the density is allreduced.
    n_fold : int, optional
        Orientational symmetry of the kernel.  Must be even.
        Default 2 (head-tail; physically correct for both rods and
        discs in the projected-axis picture).  4 → tetratic.
    """
    if n_fold % 2 != 0 or n_fold < 2:
        raise ValueError(f"n_fold must be an even integer ≥ 2 (got {n_fold})")
    m = n_fold // 2
    delta_theta = 2 * np.pi / bins
    centers = (np.arange(bins) + 0.5) * delta_theta
    diff = centers[:, None] - centers[None, :]
    W_mat = np.abs(np.sin(m * diff))

    def _cic_density(theta_local):
        t = theta_local.ravel() / delta_theta
        k = np.floor(t).astype(int) % bins
        w2 = t - np.floor(t)
        w1 = 1.0 - w2
        rho = np.zeros(bins)
        np.add.at(rho, k,              w1)
        np.add.at(rho, (k + 1) % bins, w2)
        rho = comm.allreduce(rho, op=MPI.SUM)
        rho /= (rho.sum() * delta_theta)
        return rho

    def vlasov_force(theta):
        rho = _cic_density(theta)
        W_grid = delta_theta * (W_mat @ rho)
        k_idx = (np.floor(theta.ravel() / delta_theta).astype(int)) % bins
        force = L**2 * (W_grid[k_idx] - W_grid[(k_idx + 1) % bins]) / delta_theta
        return force.reshape(-1, 1)

    def interaction_energy(theta):
        rho = _cic_density(theta)
        return float(np.sum(W_mat * rho[:, None] * rho[None, :]) * delta_theta**2)

    return vlasov_force, interaction_energy


def _disc_onsager_factory_inhomo(L, bins, bins_theta, comm, n_fold=2):
    """Build ``(vlasov_force, interaction_energy)`` for the
    *inhomogeneous* disc Onsager pair potential
    ``W = |sin(m Δθ)|`` with ``m = n_fold / 2``.

    Default ``n_fold = 2`` is the physically correct kernel for a
    3-D discotic LC simulated in a 2-D domain (θ = projected disc
    normal, head-tail symmetric).

    The Vlasov closure has signature ``vlasov_force(angle, X, density)``,
    matching the inhomogeneous transport step's calling convention
    (see ``transport_inhomo.vlasov_kick_step``).  The supplied
    ``density`` is per-particle, shape ``(nlocal, bins_theta)``: each
    row is the *probability-mass* orientation histogram of the
    particle's owning cell (rows sum to 1).  The Vlasov torque on
    particle p in θ-bin k is

        F_p = L² · (V_eff[p, k] − V_eff[p, k+1]) / Δθ,
        V_eff[p, :] = density_p · W       (no extra Δθ; mass form).

    The ``interaction_energy`` closure uses a global CIC density
    (allreduced across ranks) and is consistent with the homogeneous
    version — it tracks ``E[ρ_global]`` as a single-rank scalar
    diagnostic.

    Parameters
    ----------
    L : float
        Disc-size prefactor.
    bins : int
        Number of θ-grid cells for the CIC global density (used by
        ``interaction_energy``).
    bins_theta : int
        Number of θ-grid cells for the per-cell density supplied to
        ``vlasov_force``.  Should match ``opts["bins_theta"]`` (see
        ``CFMZNeedleDSMC._construct_grid``); the default is 32.
    comm : MPI.Comm
        Communicator for the global CIC allreduce.
    n_fold : int, optional
        Orientational symmetry; default 2.  See ``_disc_onsager_factory``.
    """
    if n_fold % 2 != 0 or n_fold < 2:
        raise ValueError(f"n_fold must be an even integer ≥ 2 (got {n_fold})")
    m = n_fold // 2

    # Per-cell kernel matrix on the bins_theta grid.
    delta_theta_cell = 2 * np.pi / bins_theta
    centers_cell = (np.arange(bins_theta) + 0.5) * delta_theta_cell
    diff_cell = centers_cell[:, None] - centers_cell[None, :]
    W_mat_cell = np.abs(np.sin(m * diff_cell))

    # Global CIC kernel matrix on the bins grid.
    delta_theta = 2 * np.pi / bins
    centers = (np.arange(bins) + 0.5) * delta_theta
    diff = centers[:, None] - centers[None, :]
    W_mat = np.abs(np.sin(m * diff))

    def _cic_density_global(theta_local):
        t = theta_local.ravel() / delta_theta
        k = np.floor(t).astype(int) % bins
        w2 = t - np.floor(t)
        w1 = 1.0 - w2
        rho = np.zeros(bins)
        np.add.at(rho, k,              w1)
        np.add.at(rho, (k + 1) % bins, w2)
        rho = comm.allreduce(rho, op=MPI.SUM)
        rho /= (rho.sum() * delta_theta)
        return rho

    def vlasov_force(angle, X, density):
        # density: (nlocal, bins_theta), per-particle owning-cell histogram
        # in **probability-mass** form (rows sum to 1; see
        # transport_inhomo._per_particle_density).  Convert to a density
        # implicitly: V_eff[k] = Σ_j (mass_j/Δθ) · W[k,j] · Δθ
        #                     = Σ_j mass_j · W[k,j].
        V_eff = density @ W_mat_cell
        theta_arr = np.asarray(angle).ravel()
        k_idx = (np.floor(theta_arr / delta_theta_cell).astype(int)) % bins_theta
        k_next = (k_idx + 1) % bins_theta
        rows = np.arange(theta_arr.size)
        force = L**2 * (V_eff[rows, k_idx] - V_eff[rows, k_next]) / delta_theta_cell
        return force.reshape(-1, 1)

    def interaction_energy(theta):
        rho = _cic_density_global(theta)
        return float(np.sum(W_mat * rho[:, None] * rho[None, :]) * delta_theta**2)

    return vlasov_force, interaction_energy


class CFMZDiscDSMCHomo(CFMZNeedleDSMCHomo):
    """DSMC solver for the homogeneous discotic CFMZ kinetic equation.

    Subclass of :class:`CFMZNeedleDSMCHomo` that swaps in the discotic
    defaults: a 4-fold-symmetric Onsager pair potential
    ``W = |sin(2(θ₁ − θ₂))|``, the tetratic order parameter family
    ``R_n`` for ``n ∈ {1, 2, 4, 6}``, and the disc-specific collision
    kernel from :mod:`dsmc.cfmz.collision_disc`.

    Constructor recognises the same ``opts`` / ``info`` keys as the
    parent, with these additional defaults applied:

    - ``opts["n_modes"]``       defaults to ``[1, 2, 4, 6]``.
    - ``info["cross_section"]`` defaults to ``"hard_disc"``.  Allowed:
      ``"maxwell"``, ``"hard_disc"`` (rigid 2-D disc; no torque
      transfer), ``"oriented_disc"`` (NTC kernel ``R |sin 2Δθ|`` with
      disc-plane contact arms; transfers ω ↔ v).
    - ``info["radius"]``        defaults to ``info.get("length", 1.0)``;
      sets the disc radius R used in the contact mechanics.
    - ``info["length"]``        is forced equal to ``info["radius"]``
      so the parent's E[ρ] half-``L²`` prefactor uses R² (the correct
      disc Onsager scale).

    .. note::

       The CFMZ kick step applies the Vlasov force directly:
       ``ω ← ω + F(θ) · dt``.  The user-supplied ``vlasov_force``
       should therefore return *angular acceleration*, not torque,
       and the integrator's energy conservation only holds when
       ``info["inertia"] = 1.0`` (the natural-units convention shared
       with all calamitic CFMZ tests).  The auto-built disc Onsager
       force respects this convention.  Set ``info["inertia"]`` to
       any other value only for diagnostics that do not depend on
       energy conservation (e.g. the Andersen-thermostat tests).

    If ``vlasov_force`` and / or ``interaction_energy`` are ``None``
    the constructor builds them automatically from ``W_disc`` using
    a CIC θ-grid of ``opts["bins"]`` cells.

    The output directory is renamed
    ``"{prefix}_output_cfmz_disc_{collision_type}"`` so that disc
    runs do not clobber needle outputs sharing the same prefix.
    """

    def __init__(
        self,
        opts: dict,
        info: dict = None,
        vlasov_force=None,
        interaction_energy=None,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ):
        info = dict(info) if info else {}
        opts = dict(opts)

        # Discotic defaults — overridable by the caller.
        # n_fold = 2 → calamitic-form kernel |sin(Δθ)|, the physically
        # correct choice for a 3-D discotic LC simulated in 2-D where θ
        # is the projected disc normal (head-tail symmetric).  Critical
        # mode is R_2 (discotic nematic, N_D).  n_fold = 4 recovers the
        # 4-fold tetratic interpretation (2-D-square-like particles).
        opts.setdefault("n_fold", 2)
        opts.setdefault("n_modes", [1, 2, 4, 6])
        info.setdefault("cross_section", "hard_disc")
        info.setdefault("radius", info.get("length", 1.0))
        # E[ρ] prefactor in the parent uses ``info["length"]``; for
        # discs the natural scale is the radius.
        info["length"] = info["radius"]

        # Auto-build disc Onsager Vlasov force / interaction energy.
        # Calling convention:
        #   None      → auto-build the disc Onsager closure with kernel
        #               W = |sin(m Δθ)|, m = n_fold//2
        #   False     → explicitly disable (parent's "no force" semantics)
        #   callable  → use the caller's closure
        bins = opts.get("bins", 31)
        n_fold = int(opts["n_fold"])
        if vlasov_force is None or interaction_energy is None:
            auto_vf, auto_ie = _disc_onsager_factory(
                info["radius"], bins, comm, n_fold=n_fold,
            )
            if vlasov_force is None:
                vlasov_force = auto_vf
            if interaction_energy is None:
                interaction_energy = auto_ie
        if vlasov_force is False:
            vlasov_force = None
        if interaction_energy is False:
            interaction_energy = None

        super().__init__(
            opts=opts,
            info=info,
            vlasov_force=vlasov_force,
            interaction_energy=interaction_energy,
            comm=comm,
        )

        # Stash n_fold so the collision module can read it (the
        # ``oriented_disc`` cross-section follows the same symmetry).
        self.n_fold = n_fold

        # Output directory — keep disc results separate from needle ones.
        self.output_path = f"{self.prefix}_output_cfmz_disc_{self.collision_type}"
        if self.rank == 0:
            os.makedirs(self.output_path, exist_ok=True)
        self.comm.Barrier()

        # Rebind the collision step to the disc-specific kernel.
        from .collision_disc import nanbu_collision_step_disc
        self.nanbu_collision_step = nanbu_collision_step_disc.__get__(self)


class CFMZDiscDSMC(CFMZNeedleDSMC):
    """DSMC solver for the **inhomogeneous** discotic CFMZ kinetic equation.

    Subclass of :class:`CFMZNeedleDSMC` (positions + orientations) that
    swaps in the discotic defaults: 4-fold-symmetric Onsager pair
    potential ``W = |sin(2(θ₁ − θ₂))|``, the tetratic order parameter
    family ``R_n`` for ``n ∈ {1, 2, 4, 6}``, the disc-specific per-cell
    collision kernel from :mod:`dsmc.cfmz.collision_disc_inhomo`, and an
    optional smectic / positional order parameter
    ``ψ_S(k) = |⟨e^{i k·x}⟩|`` (drives ``history["smectic_abs_{idx}"]``).

    The same constructor calling convention as
    :class:`CFMZDiscDSMCHomo` applies for ``vlasov_force`` /
    ``interaction_energy`` (``None`` ⇒ auto-build from ``W_disc``;
    ``False`` ⇒ explicitly disable).  Set ``opts["smectic_k"]`` to a
    list of wavevector tuples (length ``spatial_dim``) to detect
    columnar / layered structure along chosen wavevectors.

    .. note::

       The same ``info["inertia"] = 1.0`` natural-units convention as
       the other CFMZ classes applies — the Vlasov kick step does not
       divide by the moment of inertia.
    """

    def __init__(
        self,
        opts: dict,
        info: dict = None,
        vlasov_force=None,
        translational_force=None,
        interaction_energy=None,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ):
        info = dict(info) if info else {}
        opts = dict(opts)

        # Discotic defaults; see CFMZDiscDSMCHomo for the n_fold rationale.
        opts.setdefault("n_fold", 2)
        opts.setdefault("n_modes", [1, 2, 4, 6])
        info.setdefault("cross_section", "hard_disc")
        info.setdefault("radius", info.get("length", 1.0))
        info["length"] = info["radius"]

        bins = opts.get("bins", 31)
        bins_theta = opts.get("bins_theta", 32)
        n_fold = int(opts["n_fold"])
        if vlasov_force is None or interaction_energy is None:
            auto_vf, auto_ie = _disc_onsager_factory_inhomo(
                info["radius"], bins, bins_theta, comm, n_fold=n_fold,
            )
            if vlasov_force is None:
                vlasov_force = auto_vf
            if interaction_energy is None:
                interaction_energy = auto_ie
        if vlasov_force is False:
            vlasov_force = None
        if interaction_energy is False:
            interaction_energy = None

        super().__init__(
            opts=opts,
            info=info,
            vlasov_force=vlasov_force,
            translational_force=translational_force,
            interaction_energy=interaction_energy,
            comm=comm,
        )

        self.n_fold = n_fold

        # Override output path to keep disc results separate.
        self.output_path = (
            f"{self.prefix}_output_cfmz_disc_inhomo_{self.collision_type}"
        )
        if self.rank == 0:
            os.makedirs(self.output_path, exist_ok=True)
        self.comm.Barrier()

        # Rebind the collision step to the disc-specific per-cell kernel.
        from .collision_disc_inhomo import nanbu_collision_step_disc_inhomo
        self.nanbu_collision_step = nanbu_collision_step_disc_inhomo.__get__(self)
