import numpy as np

def _sample_angle(self, m: int):
    """Sample m impact angles ψ uniformly on [0, 2π)."""
    Theta = self.rng.uniform(size=(m, 1)) * 2 * np.pi
    return Theta

def nanbu_collision_step(self):
    """Rigid-rod Nanbu collision step.

    Selects Mcol = floor(nu * N * dt / 2) random collision pairs (uniform
    permutation).  For each pair:

    1. A random impact angle ψ and contact arm ℓ are drawn.
    2. The relative contact velocity V and impulse J are computed from the
       rigid-body equations with restitution coefficients ev (translational)
       and eom (rotational).
    3. Post-collisional velocities (vi', vj') and angular velocities
       (ωi', ωj') are updated in-place.

    Pairs whose orientation difference |θi - θj| < cutoff are nearly
    parallel; these fall back to a spherical-like head-on collision to
    avoid numerical instability from the near-singular denominator.
    """
    vel = self.swarm.getField("velocity").reshape(self.nlocal, self.dim)
    theta = self.swarm.getField("orientation").reshape(self.nlocal)
    omega = self.swarm.getField("angular_velocity").reshape(self.nlocal)

    # number of collision pairs
    Mcol = int(0.5 * self.nu * self.nlocal * self.dt)
    Mcol = Mcol if Mcol % 2 == 0 else Mcol - 1
    if 2 * Mcol > self.nlocal:
        Mcol = Mcol - 2
    if Mcol <= 0:
        self.swarm.restoreField("velocity")
        self.swarm.restoreField("orientation")
        self.swarm.restoreField("angular_velocity")
        return

    # random pairing: sample only the 2*Mcol indices actually needed
    pairs = self.rng.choice(self.nlocal, size=2 * Mcol, replace=False)
    i = pairs[:Mcol]
    j = pairs[Mcol:]

    # fancy indexing already produces copies, no explicit .copy() needed
    vi = vel[i]
    vj = vel[j]
    thetai = theta[i]
    thetaj = theta[j]
    omegai = omega[i]
    omegaj = omega[j]

    # rod directions
    nui = np.column_stack((np.cos(thetai), np.sin(thetai)))
    nuj = np.column_stack((np.cos(thetaj), np.sin(thetaj)))

    # random impact angle
    psi = 2.0 * np.pi * self.rng.random(Mcol)
    n = np.column_stack((np.cos(psi), np.sin(psi)))

    # cutoff for near-parallel rods
    dtheta = np.abs(thetai - thetaj)
    cutoff = self.info.get("cutoff", 0.1)
    idx = (dtheta > cutoff) & (dtheta < 2.0 * np.pi - cutoff)
    full_idx = np.where(idx)[0]
    cut_idx  = np.where(~idx)[0]

    # sampled contact arms
    L = self.info["length"]
    ell = L * self.rng.random(Mcol)

    ri = ell[:, None] * nui
    rj = L * nuj

    # relative contact velocity
    ri_perp = np.column_stack((ri[:, 1], -ri[:, 0]))
    rj_perp = np.column_stack((rj[:, 1], -rj[:, 0]))

    V = vi - vj + omegai[:, None] * ri_perp - omegaj[:, None] * rj_perp

    # impulse
    m = self.info["mass"]
    I = self.info["inertia"]

    ci = ri[:, 0] * n[:, 1] - ri[:, 1] * n[:, 0]
    cj = rj[:, 0] * n[:, 1] - rj[:, 1] * n[:, 0]

    denom = 2.0 / m + (ci**2 + cj**2) / I
    J = -np.sum(V * n, axis=1) / denom

    # restitution coefficients
    ev  = self.info.get("ev", 1.0)
    eom = self.info.get("om", 1.0)

    # pre-collision normal velocity difference (for cutoff fallback)
    vn_cut = np.sum((vi - vj) * n, axis=1)

    # apply full rigid-rod collision (non-parallel pairs)
    if full_idx.size:
        scale_v = ((1.0 + ev) * J[full_idx] / m)[:, None]
        vi[full_idx]     += scale_v * n[full_idx]
        vj[full_idx]     -= scale_v * n[full_idx]
        omegai[full_idx] -= (1.0 + eom) * J[full_idx] * ci[full_idx] / I
        omegaj[full_idx] += (1.0 + eom) * J[full_idx] * cj[full_idx] / I

    # apply spherical fallback (nearly-parallel pairs)
    if cut_idx.size:
        scale_cut = (0.5 * (1.0 + ev) * vn_cut[cut_idx])[:, None]
        vi[cut_idx] -= scale_cut * n[cut_idx]
        vj[cut_idx] += scale_cut * n[cut_idx]

    # write back
    vel[i]   = vi
    vel[j]   = vj
    omega[i] = omegai
    omega[j] = omegaj

    self.swarm.restoreField("velocity")
    self.swarm.restoreField("orientation")
    self.swarm.restoreField("angular_velocity")
