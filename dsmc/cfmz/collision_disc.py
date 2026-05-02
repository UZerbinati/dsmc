"""Disc-specific Nanbu collision kernel for the homogeneous CFMZ solver.

Two cross-section modes are supported via ``info["cross_section"]``:

``"hard_disc"`` (default)
    Equal-radius rigid 2-D discs.  Contact arms are
    ``r_i = -R n``, ``r_j = +R n`` along the contact normal ``n``.
    The lever arms ``c_k = r_k × n`` vanish identically, so the
    impulse denominator collapses to ``2/m`` and angular velocities
    are **not** modified by the collision.  Cross-section is flat
    (Maxwell-style); exactly ``Mcol = floor(ν · N · dt / 2)`` pairs
    are processed each step.

``"oriented_disc"``
    NTC acceptance with kernel ``w = |g·n| · R · |sin(2Δθ)|`` —
    the 4-fold analogue of the rod's ``L |sin Δθ|``.  Contact arms
    are biased along the in-plane disc-plane direction
    ``ν⊥ = (-sin θ, cos θ)``, giving non-zero lever arms; the full
    rigid-body impulse formula then transfers ω as well as v.

``"maxwell"`` is also accepted as a synonym for ``"hard_disc"`` —
it gives the same flat acceptance with the same hard-disc impulse.

See ``CFMZ.md`` §13 for the physics derivation.
"""
import numpy as np


def nanbu_collision_step_disc(self):
    """Hard-disc Nanbu collision step (homogeneous solver).

    Sampled identically to the rod kernel down to the pair selection;
    the **impulse mechanics** differ — hard discs do not transfer
    torque (lever arms vanish), and the oriented-disc cross-section is
    4-fold symmetric.
    """
    vel   = self.swarm.getField("velocity").reshape(self.nlocal, self.dim)
    theta = self.swarm.getField("orientation").reshape(self.nlocal)
    omega = self.swarm.getField("angular_velocity").reshape(self.nlocal)

    R = float(self.info.get("radius", self.info.get("length", 1.0)))
    cross_section = self.info.get("cross_section", "hard_disc")

    # Candidate pool size — Maxwell-style for hard_disc, NTC-style for oriented_disc.
    if cross_section == "oriented_disc":
        Mcol = int(0.5 * self._nu_max * self.nlocal * self.dt)
    else:
        Mcol = int(0.5 * self.nu * self.nlocal * self.dt)

    Mcol = min(Mcol, self.nlocal // 2)
    Mcol = Mcol if Mcol % 2 == 0 else Mcol - 1
    if Mcol <= 0:
        self.swarm.restoreField("velocity")
        self.swarm.restoreField("orientation")
        self.swarm.restoreField("angular_velocity")
        return

    pairs = self.rng.choice(self.nlocal, size=2 * Mcol, replace=False)
    i = pairs[:Mcol]
    j = pairs[Mcol:]

    vi     = vel[i]
    vj     = vel[j]
    thetai = theta[i]
    thetaj = theta[j]
    omegai = omega[i]
    omegaj = omega[j]

    # Random impact normal (uniformly sampled on the unit circle).
    psi = 2.0 * np.pi * self.rng.random(Mcol)
    n   = np.column_stack((np.cos(psi), np.sin(psi)))

    m  = self.info["mass"]
    Iz = self.info["inertia"]
    ev  = self.info.get("ev",  1.0)
    eom = self.info.get("om",  1.0)

    if cross_section in ("hard_disc", "maxwell"):
        # ------------------------------------------------------------ #
        # Spherical hard-disc impulse: lever arms = 0, no torque       #
        # transfer.  J = -(1+ev) m (V·n) / 2.                          #
        # ------------------------------------------------------------ #
        Vn = np.sum((vi - vj) * n, axis=1)            # (V_translational · n)
        scale_v = (0.5 * (1.0 + ev) * Vn)[:, None]
        vi -= scale_v * n
        vj += scale_v * n
        # ω unchanged.
        accept_idx = slice(None)                       # (book-keeping only)

    elif cross_section == "oriented_disc":
        # ------------------------------------------------------------ #
        # Orientation-coupled disc: contact arm along the disc plane,  #
        # lever arms non-zero, rigid-body impulse formula.  NTC accept #
        # with weight |g·n| · R · |sin 2Δθ|.                           #
        # ------------------------------------------------------------ #
        nui_perp = np.column_stack((-np.sin(thetai),  np.cos(thetai)))
        nuj_perp = np.column_stack((-np.sin(thetaj),  np.cos(thetaj)))
        ri = R * nui_perp
        rj = R * nuj_perp

        ri_perp = np.column_stack(( ri[:, 1], -ri[:, 0]))
        rj_perp = np.column_stack(( rj[:, 1], -rj[:, 0]))

        V = (vi - vj) + omegai[:, None] * ri_perp - omegaj[:, None] * rj_perp

        ci = ri[:, 0] * n[:, 1] - ri[:, 1] * n[:, 0]
        cj = rj[:, 0] * n[:, 1] - rj[:, 1] * n[:, 0]

        # Cross-section follows the kernel symmetry:
        # m = n_fold // 2  → S(Δθ) = R · |sin(m Δθ)|.
        # n_fold = 2 (default, head-tail) → m = 1, the calamitic
        # |sin(Δθ)| form physically correct for 3-D discotic LCs.
        # n_fold = 4 (tetratic / 2-D-square) → m = 2.
        m_kern = getattr(self, "n_fold", 2) // 2
        gn = np.abs(np.sum(V * n, axis=1))
        S  = R * np.abs(np.sin(m_kern * (thetai - thetaj)))
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

    vel[i]   = vi
    vel[j]   = vj
    omega[i] = omegai
    omega[j] = omegaj

    self.swarm.restoreField("velocity")
    self.swarm.restoreField("orientation")
    self.swarm.restoreField("angular_velocity")
