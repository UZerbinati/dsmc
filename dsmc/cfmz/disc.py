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

from . import CFMZNeedleDSMCHomo


def _disc_onsager_factory(L, bins, comm):
    """Build ``(vlasov_force, interaction_energy)`` closures for the
    disc Onsager pair potential ``W = |sin(2 Δθ)|``.

    Mirrors the kernel/CIC machinery used in ``tests/cfmz/test_12.py``
    for the rod Onsager potential.  The closures are pure functions of
    the local orientation array; MPI reductions happen inside
    ``_cic_density`` via the supplied communicator.

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
    """
    delta_theta = 2 * np.pi / bins
    centers = (np.arange(bins) + 0.5) * delta_theta
    diff = centers[:, None] - centers[None, :]
    W_mat = np.abs(np.sin(2 * diff))

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
        opts.setdefault("n_modes", [1, 2, 4, 6])
        info.setdefault("cross_section", "hard_disc")
        info.setdefault("radius", info.get("length", 1.0))
        # E[ρ] prefactor in the parent uses ``info["length"]``; for
        # discs the natural scale is the radius.
        info["length"] = info["radius"]

        # Auto-build disc Onsager Vlasov force / interaction energy.
        # Calling convention:
        #   None      → auto-build the disc Onsager closure
        #   False     → explicitly disable (parent's "no force" semantics)
        #   callable  → use the caller's closure
        bins = opts.get("bins", 31)
        if vlasov_force is None or interaction_energy is None:
            auto_vf, auto_ie = _disc_onsager_factory(info["radius"], bins, comm)
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

        # Output directory — keep disc results separate from needle ones.
        self.output_path = f"{self.prefix}_output_cfmz_disc_{self.collision_type}"
        if self.rank == 0:
            os.makedirs(self.output_path, exist_ok=True)
        self.comm.Barrier()

        # Rebind the collision step to the disc-specific kernel.
        from .collision_disc import nanbu_collision_step_disc
        self.nanbu_collision_step = nanbu_collision_step_disc.__get__(self)
