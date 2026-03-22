import numpy as np
from dsmc.utils import build_cell_lists, get_particle_cells


def nanbu_collision_step(self):
    """Nanbu collision step for Maxwell molecules (per-cell).

    Within each spatial cell the number of collision pairs is
        Mcol = floor(nu * n * dt / 2),
    where n is the local particle count.  For Maxwell molecules the collision
    kernel is independent of the relative speed, so pairs are drawn by uniform
    random permutation.  Post-collisional velocities conserve momentum and
    kinetic energy via isotropic scattering: the relative speed is preserved
    but its direction is randomised uniformly on the unit circle.
    """
    cells = get_particle_cells(self)
    cell_lists = build_cell_lists(cells)
    vel = self.swarm.getField("velocity")

    for cell, plist in cell_lists.items():
        n = len(plist)
        if n < 2:
            continue

        # Number of collision pairs; keep even and within available particles.
        Mcol = int(0.5 * self.nu * n * self.dt)
        Mcol = Mcol if Mcol % 2 == 0 else Mcol - 1
        if 2 * Mcol > n:
            Mcol = Mcol - 2
        if Mcol <= 0:
            continue

        # Select 2*Mcol distinct particles and split into partner lists i, j.
        pairs = self.rng.choice(plist, 2 * Mcol, replace=False)
        i, j = pairs[:Mcol], pairs[Mcol:]

        vi = vel[i].copy()
        vj = vel[j].copy()

        G      = 0.5 * (vi + vj)           # centre-of-mass velocity
        g_norm = np.linalg.norm(vi - vj, axis=1)  # relative speed (conserved)

        # Isotropic scattering: post-collisional relative velocity has the same
        # magnitude but a uniformly random direction on the unit circle.
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
