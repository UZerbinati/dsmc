import numpy as np

def _sample_angle(self, m: int):
    """Sample m impact angles ψ uniformly on [0, 2π)."""
    Theta = self.rng.uniform(size=(m, 1)) * 2 * np.pi
    return Theta

def nanbu_collision_step(self):
    """Rigid-rod Nanbu collision step.

    Two collision kernels are supported, selected by
    ``info["cross_section"]`` (default ``"maxwell"``):

    **Maxwell molecules** (``"maxwell"``)
        All pairs are equally likely to collide.  Exactly
        ``Mcol = floor(ν·N·dt/2)`` pairs are selected and processed
        every step.  This is the classical Nanbu method.

    **Hard needles** (``"hard_needle"``)
        The collision kernel for 2-D calamitic needles derived in
        Example B of arXiv:2508.10744 is

            W(Ξ₁, Ξ₂ → Ξ'₁, Ξ'₂) = |g·n| · S(ν₁, ν₂)

        where **g** is the effective contact velocity (relative
        translational velocity plus angular-velocity arm contributions,
        eq. 4.5 of the paper) and

            S(ν₁, ν₂) = L |sin(θ₁ − θ₂)|

        is the Onsager excluded-volume cross-section for 2-D needles.
        Near-parallel rods (|sin(Δθ)| ≈ 0) have a vanishingly small
        cross-section and are almost never selected, in sharp contrast
        to the uniform Maxwell kernel.

        Bird's **No-Time-Counter (NTC)** acceptance–rejection method is
        used:

        1. Draw ``Mcol_cand = floor(ν_max·N·dt/2)`` candidate pairs
           uniformly at random.
        2. For each candidate pair sample a contact normal **n** and arm
           ℓ, then compute

               w = |g·n| · L |sin(Δθ)|.

        3. Accept the pair with probability ``w / ν_max``.
        4. Update the running maximum: ``ν_max ← max(ν_max, max(w))``.

        The running maximum ``self._nu_max`` is initialised to ``nu``
        (the value supplied in ``opts``) and grows monotonically as the
        simulation progresses.  The user should supply a ``nu`` that is
        a reasonable first estimate of ``max |g·n| · L`` for the
        initial velocity distribution (e.g. ``nu ≈ L · v_max`` where
        ``v_max`` is the expected maximum relative speed).

    For each *accepted* pair (regardless of the cross-section model)
    the following steps are carried out:

    1. A random impact angle ψ and contact arm ℓ are drawn.
    2. The relative contact velocity **V** and impulse J are computed
       from the rigid-body collision equations with restitution
       coefficients ``ev`` (translational) and ``eom`` (rotational).
    3. Post-collisional velocities (vᵢ′, vⱼ′) and angular velocities
       (ωᵢ′, ωⱼ′) are updated in-place.

    Pairs whose orientation difference |θᵢ − θⱼ| < ``cutoff`` are
    nearly parallel; for these the full rigid-rod denominator is
    near-singular so a spherical-like head-on fallback is used instead.
    For the hard-needle kernel these pairs are already suppressed by the
    ``|sin(Δθ)|`` factor, so the fallback is rarely reached.
    """
    vel   = self.swarm.getField("velocity").reshape(self.nlocal, self.dim)
    theta = self.swarm.getField("orientation").reshape(self.nlocal)
    omega = self.swarm.getField("angular_velocity").reshape(self.nlocal)

    L             = self.info["length"]
    cross_section = self.info.get("cross_section", "maxwell")

    # ------------------------------------------------------------------ #
    # Determine candidate pool size                                        #
    # ------------------------------------------------------------------ #
    if cross_section == "maxwell":
        Mcol = int(0.5 * self.nu * self.nlocal * self.dt)
    else:  # hard_needle — NTC: pool based on running maximum kernel value
        Mcol = int(0.5 * self._nu_max * self.nlocal * self.dt)

    Mcol = min(Mcol, self.nlocal // 2)
    Mcol = Mcol if Mcol % 2 == 0 else Mcol - 1
    if Mcol <= 0:
        self.swarm.restoreField("velocity")
        self.swarm.restoreField("orientation")
        self.swarm.restoreField("angular_velocity")
        return

    # ------------------------------------------------------------------ #
    # Random pairing                                                       #
    # ------------------------------------------------------------------ #
    pairs = self.rng.choice(self.nlocal, size=2 * Mcol, replace=False)
    i = pairs[:Mcol]
    j = pairs[Mcol:]

    # fancy indexing produces copies — no explicit .copy() needed
    vi     = vel[i]
    vj     = vel[j]
    thetai = theta[i]
    thetaj = theta[j]
    omegai = omega[i]
    omegaj = omega[j]

    # rod unit vectors
    nui = np.column_stack((np.cos(thetai), np.sin(thetai)))
    nuj = np.column_stack((np.cos(thetaj), np.sin(thetaj)))

    # random impact angle ψ → contact normal n
    psi = 2.0 * np.pi * self.rng.random(Mcol)
    n   = np.column_stack((np.cos(psi), np.sin(psi)))

    # near-parallel cutoff (used for both acceptance and fallback)
    dtheta = np.abs(thetai - thetaj)
    cutoff = self.info.get("cutoff", 0.1)
    non_parallel = (dtheta > cutoff) & (dtheta < 2.0 * np.pi - cutoff)

    # sampled contact arms
    ell    = L * self.rng.random(Mcol)
    ri     = ell[:, None] * nui
    rj     = L * nuj

    # relative contact velocity  g = (p₁-p₂)/m + ω-arm contributions
    ri_perp = np.column_stack(( ri[:, 1], -ri[:, 0]))
    rj_perp = np.column_stack(( rj[:, 1], -rj[:, 0]))
    V = vi - vj + omegai[:, None] * ri_perp - omegaj[:, None] * rj_perp

    # ------------------------------------------------------------------ #
    # NTC acceptance–rejection (hard-needle kernel only)                  #
    # ------------------------------------------------------------------ #
    if cross_section == "hard_needle":
        # kernel: W = |g·n| · S,  S = L |sin(Δθ)|  (Onsager cross-section)
        gn = np.abs(np.sum(V * n, axis=1))          # |g·n|
        S  = L * np.abs(np.sin(thetai - thetaj))    # excluded-volume factor
        w  = gn * S

        w_max = float(w.max()) if w.size > 0 else 0.0
        if w_max > self._nu_max:
            self._nu_max = w_max

        accept       = self.rng.random(Mcol) < (w / self._nu_max)
        full_idx     = np.where(non_parallel & accept)[0]
        cut_idx      = np.where(~non_parallel & accept)[0]
    else:
        full_idx = np.where(non_parallel)[0]
        cut_idx  = np.where(~non_parallel)[0]

    # ------------------------------------------------------------------ #
    # Rigid-body impulse                                                  #
    # ------------------------------------------------------------------ #
    m = self.info["mass"]
    I = self.info["inertia"]

    ci = ri[:, 0] * n[:, 1] - ri[:, 1] * n[:, 0]
    cj = rj[:, 0] * n[:, 1] - rj[:, 1] * n[:, 0]

    denom = 2.0 / m + (ci**2 + cj**2) / I
    J     = -np.sum(V * n, axis=1) / denom

    ev  = self.info.get("ev",  1.0)
    eom = self.info.get("om",  1.0)

    vn_cut = np.sum((vi - vj) * n, axis=1)   # used by the fallback

    # -- full rigid-rod collision (non-parallel accepted pairs) ----------
    if full_idx.size:
        scale_v = ((1.0 + ev) * J[full_idx] / m)[:, None]
        vi[full_idx]     += scale_v * n[full_idx]
        vj[full_idx]     -= scale_v * n[full_idx]
        omegai[full_idx] -= (1.0 + eom) * J[full_idx] * ci[full_idx] / I
        omegaj[full_idx] += (1.0 + eom) * J[full_idx] * cj[full_idx] / I

    # -- spherical fallback (nearly-parallel accepted pairs) -------------
    if cut_idx.size:
        scale_cut = (0.5 * (1.0 + ev) * vn_cut[cut_idx])[:, None]
        vi[cut_idx] -= scale_cut * n[cut_idx]
        vj[cut_idx] += scale_cut * n[cut_idx]

    # ------------------------------------------------------------------ #
    # Write back                                                           #
    # ------------------------------------------------------------------ #
    vel[i]   = vi
    vel[j]   = vj
    omega[i] = omegai
    omega[j] = omegaj

    self.swarm.restoreField("velocity")
    self.swarm.restoreField("orientation")
    self.swarm.restoreField("angular_velocity")


def andersen_thermostat_step(self):
    """Andersen thermostat: randomly reset particle velocities to T_bath.

    Each particle independently, with probability ``nu_bath * dt`` per
    step, has its translational velocity **v** and angular velocity ω
    resampled from the Maxwellian at the target temperature ``T_bath``:

        v  ~ N(0, sqrt(T_bath / m))   (each Cartesian component)
        ω  ~ N(0, sqrt(T_bath / I))

    This implements the canonical heat bath of Andersen (1980), which
    ensures the invariant measure is the NVT (canonical) ensemble at
    ``T_bath``.  Orientations θ are not modified.

    The step is a no-op when ``self.T_bath`` is ``None``.
    """
    if self.T_bath is None:
        return
    m = self.info["mass"]
    I = self.info["inertia"]
    vel   = self.swarm.getField("velocity").reshape(self.nlocal, self.dim)
    omega = self.swarm.getField("angular_velocity").reshape(self.nlocal)

    mask = self.rng.random(self.nlocal) < self.nu_bath * self.dt
    n_reset = int(mask.sum())
    if n_reset > 0:
        vel[mask]   = self.rng.normal(0.0, np.sqrt(self.T_bath / m), (n_reset, self.dim))
        omega[mask] = self.rng.normal(0.0, np.sqrt(self.T_bath / I), n_reset)

    self.swarm.restoreField("velocity")
    self.swarm.restoreField("angular_velocity")
