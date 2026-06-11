"""Per-cell rigid-rod Nanbu collisions for the non-homogeneous CFMZ solver.

The pair-impulse math is identical to ``dsmc/cfmz/collision.py``; only the
particle pool is restricted to particles within a single spatial cell.  We
reproduce the kernel here rather than refactor the homogeneous file to keep
``CFMZNeedleDSMCHomo`` byte-for-byte unchanged.
"""
import numpy as np
from dsmc.utils import build_cell_lists, get_particle_cells


def _nanbu_pair_kernel(self, idxs, vel, theta, omega):
    """Run rigid-rod Nanbu on the particles whose local indices are ``idxs``.

    The state arrays ``vel`` (nlocal, dim), ``theta`` (nlocal,), and
    ``omega`` (nlocal,) are modified in-place at those indices.
    """
    n_cell = idxs.size
    L = self.info["length"]
    cross_section = self.info.get("cross_section", "maxwell")

    if cross_section == "maxwell":
        Mcol = int(0.5 * self.nu * n_cell * self.dt)
    else:
        Mcol = int(0.5 * self._nu_max * n_cell * self.dt)

    Mcol = min(Mcol, n_cell // 2)
    Mcol = Mcol if Mcol % 2 == 0 else Mcol - 1
    if Mcol <= 0:
        return

    # Pick 2*Mcol distinct *local-cell* slots, then map to global indices.
    slots = self.rng.choice(n_cell, size=2 * Mcol, replace=False)
    i_slot = slots[:Mcol]
    j_slot = slots[Mcol:]
    i = idxs[i_slot]
    j = idxs[j_slot]

    vi = vel[i].copy()
    vj = vel[j].copy()
    thetai = theta[i].copy()
    thetaj = theta[j].copy()
    omegai = omega[i].copy()
    omegaj = omega[j].copy()

    nui = np.column_stack((np.cos(thetai), np.sin(thetai)))
    nuj = np.column_stack((np.cos(thetaj), np.sin(thetaj)))

    psi = 2.0 * np.pi * self.rng.random(Mcol)
    n = np.column_stack((np.cos(psi), np.sin(psi)))

    dtheta = np.abs(thetai - thetaj)
    cutoff = self.info.get("cutoff", 0.1)
    non_parallel = (dtheta > cutoff) & (dtheta < 2.0 * np.pi - cutoff)

    ell = L * self.rng.random(Mcol)
    ri = ell[:, None] * nui
    rj = L * nuj

    ri_perp = np.column_stack((ri[:, 1], -ri[:, 0]))
    rj_perp = np.column_stack((rj[:, 1], -rj[:, 0]))
    V = vi - vj + omegai[:, None] * ri_perp - omegaj[:, None] * rj_perp

    if cross_section == "hard_needle":
        # ``sin_dtheta_floor`` (default 0): regularises the thin-rod
        # cross-section so parallel rods still collide with weight
        # ≈ L·sin_dtheta_floor.  See collision_enskog_inhomo.py for
        # the rationale (Vlasov-pinned R₂≈1 case).
        sin_floor = self.info.get("sin_dtheta_floor", 0.0)
        gn = np.abs(np.sum(V * n, axis=1))
        S = L * np.maximum(np.abs(np.sin(thetai - thetaj)), sin_floor)
        w = gn * S
        w_max = float(w.max()) if w.size > 0 else 0.0
        if w_max > self._nu_max:
            self._nu_max = w_max
        accept = self.rng.random(Mcol) < (w / self._nu_max)
        full_idx = np.where(non_parallel & accept)[0]
        cut_idx = np.where(~non_parallel & accept)[0]
    else:
        full_idx = np.where(non_parallel)[0]
        cut_idx = np.where(~non_parallel)[0]

    m = self.info["mass"]
    I = self.info["inertia"]

    ci = ri[:, 0] * n[:, 1] - ri[:, 1] * n[:, 0]
    cj = rj[:, 0] * n[:, 1] - rj[:, 1] * n[:, 0]

    denom = 2.0 / m + (ci**2 + cj**2) / I
    J = -np.sum(V * n, axis=1) / denom

    ev = self.info.get("ev", 1.0)
    eom = self.info.get("om", 1.0)

    vn_cut = np.sum((vi - vj) * n, axis=1)

    if full_idx.size:
        scale_v = ((1.0 + ev) * J[full_idx] / m)[:, None]
        vi[full_idx] += scale_v * n[full_idx]
        vj[full_idx] -= scale_v * n[full_idx]
        omegai[full_idx] -= (1.0 + eom) * J[full_idx] * ci[full_idx] / I
        omegaj[full_idx] += (1.0 + eom) * J[full_idx] * cj[full_idx] / I

    if cut_idx.size:
        scale_cut = (0.5 * (1.0 + ev) * vn_cut[cut_idx])[:, None]
        vi[cut_idx] -= scale_cut * n[cut_idx]
        vj[cut_idx] += scale_cut * n[cut_idx]

    vel[i] = vi
    vel[j] = vj
    theta[i] = thetai
    theta[j] = thetaj
    omega[i] = omegai
    omega[j] = omegaj


def nanbu_collision_step(self):
    """Per-cell rigid-rod Nanbu collision step."""
    if self.nlocal == 0:
        return

    cells = get_particle_cells(self)
    cell_lists = build_cell_lists(cells)

    vel = self.swarm.getField("velocity").reshape(self.nlocal, self.dim)
    theta = self.swarm.getField("orientation").reshape(self.nlocal)
    omega = self.swarm.getField("angular_velocity").reshape(self.nlocal)

    for cell_id, idxs in cell_lists.items():
        if idxs.size < 2:
            continue
        _nanbu_pair_kernel(self, idxs, vel, theta, omega)

    self.swarm.restoreField("velocity")
    self.swarm.restoreField("orientation")
    self.swarm.restoreField("angular_velocity")


def andersen_thermostat_step(self):
    """Andersen thermostat — per-particle, ignores cell structure."""
    if self.T_bath is None or self.nlocal == 0:
        return
    m = self.info["mass"]
    I = self.info["inertia"]
    vel = self.swarm.getField("velocity").reshape(self.nlocal, self.dim)
    omega = self.swarm.getField("angular_velocity").reshape(self.nlocal)

    mask = self.rng.random(self.nlocal) < self.nu_bath * self.dt
    n_reset = int(mask.sum())
    if n_reset > 0:
        vel[mask] = self.rng.normal(0.0, np.sqrt(self.T_bath / m), (n_reset, self.dim))
        omega[mask] = self.rng.normal(0.0, np.sqrt(self.T_bath / I), n_reset)

    self.swarm.restoreField("velocity")
    self.swarm.restoreField("angular_velocity")
