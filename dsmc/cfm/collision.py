import numpy as np
def _build_cell_lists(cells):
    if cells.size == 0:
        return {}

    order = np.argsort(cells)
    cells_sorted = cells[order]

    # starts of each new cell block
    starts = np.flatnonzero(
        np.r_[True, cells_sorted[1:] != cells_sorted[:-1]]
    )
    ends = np.r_[starts[1:], len(cells_sorted)]

    cell_lists = {}
    for a, b in zip(starts, ends):
        c = int(cells_sorted[a])
        cell_lists[c] = order[a:b]

    return cell_lists

def _sample_angle(self, m: int):
    Theta = self.rng.uniform(size=(m, 1)) * 2 * np.pi
    return Theta

def _get_particle_cells(self):
    celldm = self.swarm.getCellDMActive()
    cellid_name = celldm.getCellID()

    arr = self.swarm.getField(cellid_name)
    try:
        out = np.asarray(arr).copy()
        return out.reshape(-1).astype(np.int32)
    finally:
         self.swarm.restoreField(cellid_name)

def nanbu_collision_step(self):
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

    # random permutation and pairing
    r = self.rng.permutation(self.nlocal)
    i = r[:Mcol]
    j = r[Mcol:2 * Mcol]

    vi = vel[i].copy()
    vj = vel[j].copy()

    thetai = theta[i].copy()
    thetaj = theta[j].copy()

    omegai = omega[i].copy()
    omegaj = omega[j].copy()

    # rod directions
    nui = np.column_stack((np.cos(thetai), np.sin(thetai)))
    nuj = np.column_stack((np.cos(thetaj), np.sin(thetaj)))

    # random impact angle
    psi = 2.0 * np.pi * self.rng.random(Mcol)
    n = np.column_stack((np.cos(psi), np.sin(psi)))
    n_cutoff = n.copy()

    # cutoff for near-parallel rods
    dtheta = np.abs(thetai - thetaj)
    cutoff = self.info.get("cutoff", 0.1)
    idx = (dtheta > cutoff) & (dtheta < 2.0 * np.pi - cutoff)
    idxf = idx.astype(float)
    not_idxf = 1.0 - idxf

    # sampled contact arms
    L = self.info["length"]
    ell = L * self.rng.random(Mcol)

    ri = ell[:, None] * nui
    rj = L * nuj   # this matches your MATLAB code literally

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
    ev = self.info.get("ev", 1.0)
    eom = self.info.get("om", 1.0)

    # fallback spherical-like collision for nearly parallel rods
    vn_cut = np.sum((vi - vj) * n_cutoff, axis=1)

    vi_prime = (
        vi
        + ((1.0 + ev) * J / m)[:, None] * n * idxf[:, None]
        - 0.5 * (1.0 + ev) * vn_cut[:, None] * n_cutoff * not_idxf[:, None]
    )

    vj_prime = (
        vj
        - ((1.0 + ev) * J / m)[:, None] * n * idxf[:, None]
        + 0.5 * (1.0 + ev) * vn_cut[:, None] * n_cutoff * not_idxf[:, None]
    )

    omegai_prime = omegai - (1.0 + eom) * J * ci / I * idxf
    omegaj_prime = omegaj + (1.0 + eom) * J * cj / I * idxf

    # write back
    vel[i] = vi_prime
    vel[j] = vj_prime
    omega[i] = omegai_prime
    omega[j] = omegaj_prime

    self.swarm.restoreField("velocity")
    self.swarm.restoreField("orientation")
    self.swarm.restoreField("angular_velocity")
