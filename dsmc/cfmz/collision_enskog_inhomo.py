"""
Enskog-DSMC collision kernel for the inhomogeneous CFMZ needle solver.

This module supplies the Enskog generalisation of the per-cell rod
Nanbu kernel (``collision_inhomo.py``).  Boltzmann's collision integral
factors out positions and treats every collision as local; Enskog
restores both the *non-locality* of contact (centres separated by the
contact vector) and the **density-dependent collision rate** captured
by the contact value of the pair correlation function ``g(σ⁺)``.  The
resulting kinetic equation is the natural setting for emergent
positional ordering — and in particular the smectic-A phase that hard
rods develop above some packing fraction (Bates & Frenkel 1998 *J.
Chem. Phys.* 109:6193 in 2-D; Bolhuis & Frenkel 1997 PRE 56:5495 in
3-D).

This file is the calamitic-rod counterpart of CFMZ.md §14.

Physics summary
===============

Boltzmann pair-collision rate for two particles in cell C:

    ν_B(ξ_1, ξ_2) ∝ |g_rel · n̂| · σ(ξ_1, ξ_2)

with ``n̂`` the (uniformly random) impact normal and σ the
orientational cross-section — for 2-D thin rods, the Onsager form
``L |sin Δθ|``.  Pair selection in Boltzmann DSMC is uniform inside
each spatial cell (``collision_inhomo._nanbu_pair_kernel``).

Enskog modifies this in three places:

1. **Contact normal from positions**.  Instead of sampling ``n̂`` uniformly
   on the unit circle, set

       n̂_ij = (x_i − x_j) / ‖x_i − x_j‖.

   This is what introduces position-orientation coupling: the same
   rigid-body impulse formula now depends on where the two particles
   actually are, not just their velocities and orientations.  In
   aggregate, this drives the position-correlated structures (smectic
   stripes) that Boltzmann sampling cannot produce.

2. **Cross-cell candidate pool**.  Two particles whose centres are
   within the contact distance (≈ rod length L) but in different
   spatial cells are valid collision partners — and in fact contribute
   most of the surviving pairs at high density.  The candidate pool
   for each cell C is therefore C ∪ (its 8 neighbours in 2-D, 2 in
   1-D), with periodic wrap-around when the BCs are periodic.

3. **Parsons-Lee density correction**.  Multiply the NTC acceptance
   weight by

       g_PL(η) = (1 − η/4) / (1 − η)²,    η = (π/4) ρ_local L²

   the rod-equivalent of the Carnahan-Starling contact correlation
   function (Parsons 1979; Lee 1987/1988).  In the dilute limit
   ``η → 0`` we get ``g_PL → 1`` and recover Boltzmann DSMC; near
   freezing (η → 1) the factor diverges, capturing the rapid
   crowding-induced collision-rate enhancement.  Above η = 1 the
   factor is undefined; we clamp to a finite cap to avoid blow-up
   without raising — the user is then in unphysically dense territory
   and should be aware of it.

Algorithm (Frezzotti-style, simplified for 2-D rods)
====================================================

For each spatial cell C:
1. Build the candidate pool ``P_C = C ∪ neighbours(C)`` from the
   pre-computed cell list (neighbours are looked up via the global
   cell index ``i + j·nx``; periodic wrap-around in the index lookup).
2. NTC sampling.  Draw ``M_cand = ⌊½ ν_max |C| Δt⌋`` candidate pairs
   ``(i, j)`` with ``i ∈ C`` and ``j ∈ P_C \\ {i}``.  The "i ∈ C"
   restriction is what avoids double-counting cross-cell pairs (each
   pair is processed exactly once, when the loop visits the cell
   containing i).
3. For each candidate pair:
   - r_ij = x_i - x_j; reject pairs with ‖r_ij‖ > L_cut (taken to be L)
     to spare cycles on physically irrelevant pairs.
   - n̂_ij = r_ij / ‖r_ij‖.
   - Sample contact arms uniformly along the rods, just like the
     Boltzmann kernel: ``r_i = ℓ_i · ν_i`` with ``ℓ_i ∼ U(0, L)``,
     ``r_j = L · ν_j``.  This preserves the rigid-rod impulse
     mechanics with non-zero lever arms (so ω is exchanged through
     collisions, unlike the disc case).
   - Effective contact velocity
     ``V = (v_i − v_j) + ω_i r_i⊥ − ω_j r_j⊥``.
   - Acceptance weight
     ``w = |V · n̂_ij| · L|sin Δθ| · g_PL(η_C)``.
   - Accept with probability ``w / ν_max``; update ``ν_max``.
4. For accepted pairs apply the rigid-rod impulse from
   ``collision_inhomo._nanbu_pair_kernel`` (lines 64-99) using
   ``n̂_ij`` as the contact normal.  The post-collision update
   propagates v and ω in place.

Cell-sizing constraint
======================

Step (3) only finds collision partners within the cell ∪ neighbours
neighbourhood, so the spatial cell width must be ≳ L:

    L_x / bins  ≥  L     (and similarly L_y / bins for 2-D).

The constructor of ``CFMZNeedleDSMC`` warns when this is violated.
For physically reasonable Enskog runs at ``η ∼ 0.4`` the user picks
``bins`` and ``L`` such that ``ρ L² ∼ O(1)`` and the cell width is at
least L; see CFMZ.md §14.4 for parameter guidance.

MPI
===

Cross-rank ghost-particle migration is *not* implemented: the cell
neighbours that lie on a different MPI rank are silently dropped from
the pool.  This is acceptable for serial / few-rank runs (mpirun -n 1
to ~4); the constructor emits a warning when a higher rank count is
detected.  Multi-rank Enskog is a research follow-up.

References
==========
- C. Cercignani, *The Boltzmann Equation and Its Applications* (1988),
  §3.6 for the Enskog correction.
- A. Frezzotti, *Physica A* 240:202 (1997); *Eur. J. Mech. B* 17:651
  (1998) — DSMC for hard-sphere Enskog.
- N. F. Carnahan & K. E. Starling, *J. Chem. Phys.* 51:635 (1969).
- J. D. Parsons, *Phys. Rev. A* 19:1225 (1979).
- S. Lee, *J. Chem. Phys.* 87:4972 (1987); *J. Chem. Phys.* 89:7036 (1988).
- M. P. Allen, G. T. Evans, D. Frenkel & B. Mulder, *Adv. Chem. Phys.*
  86:1 (1993) for the Onsager virial.
- M. A. Bates & D. Frenkel, *J. Chem. Phys.* 109:6193 (1998) for the
  2-D smectic-A reference simulation.
"""
import numpy as np
from mpi4py import MPI

from dsmc.utils import build_cell_lists


def _parsons_lee(eta, eta_cap=0.85):
    """Parsons-Lee contact correlation factor for 2-D rods.

    Returns the multiplicative correction to the Boltzmann collision
    rate that accounts for finite-size crowding:

        g_PL(η) = (1 − η/4) / (1 − η)².

    Clamped to η ≤ ``eta_cap`` (default 0.85) so the formula does not
    blow up in unphysically dense corners; in practice
    ``η ≳ 0.7`` is already in the smectic / crystalline regime where
    Onsager-Parsons-Lee mean-field theory becomes inaccurate.

    Parameters
    ----------
    eta : float or np.ndarray
        Local rod packing fraction ``(π/4) ρ L²``.
    eta_cap : float, optional
        Hard cap on the input.

    Returns
    -------
    g : same shape as ``eta``.
    """
    eta_eff = np.minimum(eta, eta_cap)
    return (1.0 - 0.25 * eta_eff) / (1.0 - eta_eff) ** 2


def _global_cell_index(self, X):
    """Compute integer cell index ``i + j·nx`` per particle from positions.

    Avoids the DMSwarm cell-ID dependency (which would be MPI-local
    and partition-dependent).  The index is computed on the *global*
    grid so neighbour lookups across rank boundaries can be expressed
    consistently — though we do not actually fetch particles across
    ranks (see module docstring, MPI section).

    Parameters
    ----------
    X : (nlocal, mesh_dim) array
        Particle positions.

    Returns
    -------
    cell_id : (nlocal,) int64 array of global cell indices in [0, nx*ny).
    nx, ny : grid dimensions
    cell_volume : float (length in 1-D, area in 2-D)
    """
    bins = self.bins
    if self.spatial_dim == 1:
        Lx = self.info["Lx"]
        dx = Lx / bins
        i_idx = np.clip((X[:, 0] / dx).astype(np.int64), 0, bins - 1)
        return i_idx, bins, 1, dx
    else:
        xmin = self.info["xmin"]
        ymin = self.info["ymin"]
        xmax = self.info["xmax"]
        ymax = self.info["ymax"]
        dx = (xmax - xmin) / bins
        dy = (ymax - ymin) / bins
        i_idx = np.clip(((X[:, 0] - xmin) / dx).astype(np.int64), 0, bins - 1)
        j_idx = np.clip(((X[:, 1] - ymin) / dy).astype(np.int64), 0, bins - 1)
        return i_idx + j_idx * bins, bins, bins, dx * dy


def _neighbour_cell_ids(cell_id, nx, ny, periodic):
    """Return the list of global neighbour-cell IDs for ``cell_id``.

    Includes the cell itself.  In 2-D returns up to 9 IDs; in 1-D
    (ny == 1) returns up to 3.

    Parameters
    ----------
    cell_id : int
        Global cell index ``i + j·nx``.
    nx, ny : int
        Grid dimensions.  ny=1 means 1-D.
    periodic : bool
        If True, wrap indices around the box.  If False, drop
        out-of-range neighbours.
    """
    i = cell_id % nx
    j = cell_id // nx
    out = []
    for dj in (-1, 0, 1) if ny > 1 else (0,):
        for di in (-1, 0, 1):
            ii = i + di
            jj = j + dj
            if periodic:
                ii %= nx
                jj %= ny
            else:
                if ii < 0 or ii >= nx or jj < 0 or jj >= ny:
                    continue
            out.append(ii + jj * nx)
    return out


def nanbu_collision_step_enskog_inhomo(self):
    """Per-cell Enskog Nanbu collision step.

    See module docstring for the math; see
    ``collision_inhomo._nanbu_pair_kernel`` for the rigid-rod impulse
    formula reused here.
    """
    if self.nlocal == 0:
        return

    # Read all the fields we need.
    celldm = self.swarm.getCellDMActive()
    coord_names = celldm.getCoordinateFields()
    pos = self.swarm.getField(coord_names[0])
    vel = self.swarm.getField("velocity").reshape(self.nlocal, self.dim)
    theta = self.swarm.getField("orientation").reshape(self.nlocal)
    omega = self.swarm.getField("angular_velocity").reshape(self.nlocal)

    try:
        X = np.asarray(pos).reshape(self.nlocal, self.mesh_dim)
        L = self.info["length"]
        ev = self.info.get("ev", 1.0)
        eom = self.info.get("om", 1.0)
        cutoff = self.info.get("cutoff", 0.1)
        bcs = self.info.get("bcs", "periodic")
        periodic = (bcs == "periodic")

        # Compute per-particle global cell id.
        cell_id, nx, ny, cell_volume = _global_cell_index(self, X)

        # Build cell list once.
        cell_lists = build_cell_lists(cell_id)

        for cid, idxs_local in cell_lists.items():
            n_local = idxs_local.size
            if n_local < 1:
                continue

            # Pool: local cell + neighbours.
            neigh_ids = _neighbour_cell_ids(cid, nx, ny, periodic)
            pools = [cell_lists[nid] for nid in neigh_ids if nid in cell_lists]
            if not pools:
                continue
            pool = np.concatenate(pools)
            n_pool = pool.size

            # Local packing fraction & Parsons-Lee correction.
            rho_local = n_local / cell_volume
            eta_local = 0.25 * np.pi * rho_local * L * L
            g_PL = float(_parsons_lee(eta_local))

            # NTC candidate count, scaled by local rate so that no
            # cross-cell double-counting can occur (i ∈ local only).
            Mcol = int(0.5 * self._nu_max * n_local * self.dt)
            Mcol = min(Mcol, n_local, n_pool - 1)
            if Mcol <= 0:
                continue

            # Sample i from local cell (without replacement) and
            # j from the cell-pool (with replacement, ensuring i ≠ j).
            i_part = self.rng.choice(idxs_local, size=Mcol, replace=False)
            j_part = self.rng.choice(pool, size=Mcol, replace=True)
            mask = i_part != j_part
            i_part = i_part[mask]
            j_part = j_part[mask]
            if i_part.size == 0:
                continue

            # Position-derived contact normal — the Enskog feature.
            r_ij = X[i_part] - X[j_part]
            # Unwrap relative position under periodic BCs so a pair
            # that "wraps" the box still has the right contact normal.
            if periodic:
                if self.spatial_dim >= 1:
                    Lx = self.info.get("Lx",
                        self.info.get("xmax", 1.0) - self.info.get("xmin", 0.0))
                    r_ij[:, 0] -= Lx * np.round(r_ij[:, 0] / Lx)
                if self.spatial_dim >= 2:
                    Ly = self.info["ymax"] - self.info["ymin"]
                    r_ij[:, 1] -= Ly * np.round(r_ij[:, 1] / Ly)
            r_norm = np.linalg.norm(r_ij[:, : self.spatial_dim], axis=1)

            # Reject pairs at distance > L (outside the contact shell).
            close = r_norm < L
            if not np.any(close):
                continue
            i_part = i_part[close]
            j_part = j_part[close]
            r_ij_close = r_ij[close, : self.spatial_dim]
            r_norm_close = r_norm[close]
            # Padded zero in case the simulation is 1-D — n needs 2 components
            # for the rod kernel that lives in the (vx, vy) plane.
            if self.spatial_dim == 1:
                n_contact = np.column_stack(
                    (r_ij_close[:, 0] / r_norm_close, np.zeros_like(r_norm_close))
                )
            else:
                n_contact = r_ij_close / r_norm_close[:, None]

            # ---- Rigid-rod impulse (mirroring collision_inhomo) ----
            vi = vel[i_part]
            vj = vel[j_part]
            thetai = theta[i_part]
            thetaj = theta[j_part]
            omegai = omega[i_part]
            omegaj = omega[j_part]

            nui = np.column_stack((np.cos(thetai), np.sin(thetai)))
            nuj = np.column_stack((np.cos(thetaj), np.sin(thetaj)))

            # Same contact-arm sampling as the Boltzmann rod kernel:
            # arm on rod i is uniform along [0, L]; arm on rod j is
            # the tip (L · ν_j).  This preserves the rigid-rod
            # impulse content; only n̂ has been replaced.
            ell = L * self.rng.random(i_part.size)
            ri = ell[:, None] * nui
            rj = L * nuj
            ri_perp = np.column_stack((ri[:, 1], -ri[:, 0]))
            rj_perp = np.column_stack((rj[:, 1], -rj[:, 0]))

            V = vi - vj + omegai[:, None] * ri_perp - omegaj[:, None] * rj_perp

            # NTC weight.
            gn = np.abs(np.sum(V * n_contact, axis=1))
            S = L * np.abs(np.sin(thetai - thetaj))
            w = gn * S * g_PL

            if w.size > 0:
                w_max = float(w.max())
                if w_max > self._nu_max:
                    self._nu_max = w_max

            accept = self.rng.random(i_part.size) < (w / max(self._nu_max, 1e-30))

            # Near-parallel cutoff (rigid-rod denominator can
            # explode); fall back to the spherical impulse there.
            dtheta = np.abs(thetai - thetaj)
            non_parallel = (dtheta > cutoff) & (dtheta < 2.0 * np.pi - cutoff)

            full_idx = np.where(accept & non_parallel)[0]
            cut_idx = np.where(accept & ~non_parallel)[0]

            m = self.info["mass"]
            Iz = self.info["inertia"]

            ci = ri[:, 0] * n_contact[:, 1] - ri[:, 1] * n_contact[:, 0]
            cj = rj[:, 0] * n_contact[:, 1] - rj[:, 1] * n_contact[:, 0]

            denom = 2.0 / m + (ci ** 2 + cj ** 2) / Iz
            J = -np.sum(V * n_contact, axis=1) / denom
            vn_cut = np.sum((vi - vj) * n_contact, axis=1)

            if full_idx.size:
                scale_v = ((1.0 + ev) * J[full_idx] / m)[:, None]
                vi[full_idx] += scale_v * n_contact[full_idx]
                vj[full_idx] -= scale_v * n_contact[full_idx]
                omegai[full_idx] -= (1.0 + eom) * J[full_idx] * ci[full_idx] / Iz
                omegaj[full_idx] += (1.0 + eom) * J[full_idx] * cj[full_idx] / Iz

            if cut_idx.size:
                scale_cut = (0.5 * (1.0 + ev) * vn_cut[cut_idx])[:, None]
                vi[cut_idx] -= scale_cut * n_contact[cut_idx]
                vj[cut_idx] += scale_cut * n_contact[cut_idx]

            # Write back the modified slices to the swarm fields.
            vel[i_part] = vi
            vel[j_part] = vj
            omega[i_part] = omegai
            omega[j_part] = omegaj

    finally:
        self.swarm.restoreField(coord_names[0])
        self.swarm.restoreField("velocity")
        self.swarm.restoreField("orientation")
        self.swarm.restoreField("angular_velocity")
