"""Per-cell hard-disc Nanbu collisions for the inhomogeneous CFMZ solver.

Same per-cell driver structure as ``collision_inhomo.py``; the per-pair
impulse mathematics is the disc version (``hard_disc`` and
``oriented_disc``) — see ``collision_disc.py`` and CFMZ.md §13.3.
"""
import numpy as np
from dsmc.utils import build_cell_lists, get_particle_cells


def _nanbu_pair_kernel_disc(self, idxs, vel, theta, omega):
    """Run hard-disc Nanbu on the particles whose local indices are ``idxs``.

    The state arrays ``vel``, ``theta``, ``omega`` are modified in place
    at those indices.

    Two cross-section modes are supported:

    ``"hard_disc"`` (default and ``"maxwell"`` synonym)
        Equal-radius rigid 2-D discs.  Lever arms vanish identically;
        ω is **not** modified by the collision.  Cross-section is flat
        Maxwell-style; exactly ``floor(ν·n_cell·dt/2)`` pairs accepted.

    ``"oriented_disc"``
        NTC acceptance with kernel ``|g·n|·R·|sin(2Δθ)|`` and
        contact arms biased along the disc-plane direction
        ``ν⊥ = (-sin θ, cos θ)``.  Lever arms non-zero ⇒ full
        rigid-body impulse, ω updated.
    """
    n_cell = idxs.size
    R = float(self.info.get("radius", self.info.get("length", 1.0)))
    cross_section = self.info.get("cross_section", "hard_disc")

    if cross_section == "oriented_disc":
        Mcol = int(0.5 * self._nu_max * n_cell * self.dt)
    else:
        Mcol = int(0.5 * self.nu * n_cell * self.dt)

    Mcol = min(Mcol, n_cell // 2)
    Mcol = Mcol if Mcol % 2 == 0 else Mcol - 1
    if Mcol <= 0:
        return

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

    psi = 2.0 * np.pi * self.rng.random(Mcol)
    n = np.column_stack((np.cos(psi), np.sin(psi)))

    m  = self.info["mass"]
    Iz = self.info["inertia"]
    ev  = self.info.get("ev",  1.0)
    eom = self.info.get("om",  1.0)

    if cross_section in ("hard_disc", "maxwell"):
        Vn = np.sum((vi - vj) * n, axis=1)
        scale_v = (0.5 * (1.0 + ev) * Vn)[:, None]
        vi -= scale_v * n
        vj += scale_v * n
        # ω is unchanged (lever arms vanish for spherical-disc contact).

    elif cross_section == "oriented_disc":
        nui_perp = np.column_stack((-np.sin(thetai),  np.cos(thetai)))
        nuj_perp = np.column_stack((-np.sin(thetaj),  np.cos(thetaj)))
        ri = R * nui_perp
        rj = R * nuj_perp

        ri_perp = np.column_stack(( ri[:, 1], -ri[:, 0]))
        rj_perp = np.column_stack(( rj[:, 1], -rj[:, 0]))

        V = (vi - vj) + omegai[:, None] * ri_perp - omegaj[:, None] * rj_perp

        ci = ri[:, 0] * n[:, 1] - ri[:, 1] * n[:, 0]
        cj = rj[:, 0] * n[:, 1] - rj[:, 1] * n[:, 0]

        gn = np.abs(np.sum(V * n, axis=1))
        S  = R * np.abs(np.sin(2.0 * (thetai - thetaj)))
        w  = gn * S
        if w.size > 0:
            w_max = float(w.max())
            if w_max > self._nu_max:
                self._nu_max = w_max
        accept_mask = self.rng.random(Mcol) < (w / max(self._nu_max, 1e-30))
        accept_idx  = np.where(accept_mask)[0]

        if accept_idx.size:
            denom = 2.0 / m + (ci[accept_idx] ** 2 + cj[accept_idx] ** 2) / Iz
            J = -np.sum(V[accept_idx] * n[accept_idx], axis=1) / denom

            scale_v = ((1.0 + ev) * J / m)[:, None]
            vi[accept_idx] += scale_v * n[accept_idx]
            vj[accept_idx] -= scale_v * n[accept_idx]
            omegai[accept_idx] -= (1.0 + eom) * J * ci[accept_idx] / Iz
            omegaj[accept_idx] += (1.0 + eom) * J * cj[accept_idx] / Iz

    else:
        raise ValueError(
            f"[!] Unknown disc cross_section: {cross_section!r}. "
            "Expected one of: 'hard_disc', 'oriented_disc', 'maxwell'."
        )

    vel[i] = vi
    vel[j] = vj
    theta[i] = thetai
    theta[j] = thetaj
    omega[i] = omegai
    omega[j] = omegaj


def nanbu_collision_step_disc_inhomo(self):
    """Per-cell hard-disc Nanbu collision step."""
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
        _nanbu_pair_kernel_disc(self, idxs, vel, theta, omega)

    self.swarm.restoreField("velocity")
    self.swarm.restoreField("orientation")
    self.swarm.restoreField("angular_velocity")
