import numpy as np


def _build_cell_lists(cells):
    if cells.size == 0:
        return {}
    order = np.argsort(cells)
    cells_sorted = cells[order]
    starts = np.flatnonzero(
        np.r_[True, cells_sorted[1:] != cells_sorted[:-1]]
    )
    ends = np.r_[starts[1:], len(cells_sorted)]
    cell_lists = {}
    for a, b in zip(starts, ends):
        c = int(cells_sorted[a])
        cell_lists[c] = order[a:b]
    return cell_lists


def _get_particle_cells(self):
    celldm = self.swarm.getCellDMActive()
    cellid_name = celldm.getCellID()
    arr = self.swarm.getField(cellid_name)
    try:
        return np.asarray(arr).reshape(-1).astype(np.int32).copy()
    finally:
        self.swarm.restoreField(cellid_name)


def nanbu_collision_step(self):
    cells = _get_particle_cells(self)
    cell_lists = _build_cell_lists(cells)
    vel = self.swarm.getField("velocity")

    for cell, plist in cell_lists.items():
        n = len(plist)
        if n < 2:
            continue

        Mcol = int(0.5 * self.nu * n * self.dt)
        Mcol = Mcol if Mcol % 2 == 0 else Mcol - 1
        if 2 * Mcol > n:
            Mcol = Mcol - 2
        if Mcol <= 0:
            continue

        pairs = self.rng.choice(plist, 2 * Mcol, replace=False)
        i, j = pairs[:Mcol], pairs[Mcol:]

        vi = vel[i].copy()
        vj = vel[j].copy()

        g = vi - vj
        G = 0.5 * (vi + vj)
        g_norm = np.linalg.norm(g, axis=1)

        # Isotropic scattering: post-collisional relative velocity has same
        # magnitude but random direction (Maxwell molecules)
        sigma = self.rng.uniform(0.0, 2.0 * np.pi, Mcol)
        vel[i, 0] = G[:, 0] + 0.5 * g_norm * np.cos(sigma)
        vel[i, 1] = G[:, 1] + 0.5 * g_norm * np.sin(sigma)
        vel[j, 0] = G[:, 0] - 0.5 * g_norm * np.cos(sigma)
        vel[j, 1] = G[:, 1] - 0.5 * g_norm * np.sin(sigma)

    self.swarm.restoreField("velocity")


def bgk_collision_step(self):
    cells = _get_particle_cells(self)
    cell_lists = _build_cell_lists(cells)
    vel = self.swarm.getField("velocity")

    for cell, plist in cell_lists.items():
        n = len(plist)
        if n < 2:
            continue

        v_cell = vel[plist].copy()
        v_mean = v_cell.mean(axis=0)
        e_mean = (v_cell ** 2).mean(axis=0)

        gaussian = self.rng.normal(size=(n, self.dim))
        # Smooth twice to match cell mean velocity and energy exactly
        for _ in range(2):
            g_mean = gaussian.mean(axis=0)
            g_e = (gaussian ** 2).mean(axis=0)
            var_g = g_e - g_mean ** 2
            var_target = e_mean - v_mean ** 2
            tau = np.sqrt(np.where(var_target > 0, var_g / var_target, 1.0))
            lam = g_mean - tau * v_mean
            gaussian = (gaussian - lam) / tau

        vel[plist] = gaussian

    self.swarm.restoreField("velocity")
