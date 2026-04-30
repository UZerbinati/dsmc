"""Transport substeps for the non-space-homogeneous CFMZ solver.

Combines Boltzmann-style spatial advection (X += V·dt + BCs + migrate) with
the orientation drift θ += ω·dt from the homogeneous CFMZ.  The Vlasov kick
applies BOTH a translational force G[f](θ, x) and an angular torque
F[f](θ, x), each evaluated against a per-cell orientation density.
"""
import numpy as np
from dsmc.utils import build_cell_lists, get_particle_cells


def _reflect_1d(x, v, xmin, xmax):
    """Elastic reflection at domain boundaries [xmin, xmax]."""
    left = x < xmin
    right = x > xmax
    x[left] = 2.0 * xmin - x[left]
    v[left] = -v[left]
    x[right] = 2.0 * xmax - x[right]
    v[right] = -v[right]


def _periodic_1d(x, xmin, xmax):
    """Periodic wrap of x onto [xmin, xmax]."""
    L = xmax - xmin
    x[:] = xmin + (x - xmin) % L


def _apply_bcs(self, X, V):
    """Apply spatial boundary conditions in-place on X (and V for reflective)."""
    bcs = self.info.get("bcs", "reflective")
    if self.spatial_dim == 1:
        Lx = self.info["Lx"]
        if bcs == "reflective":
            _reflect_1d(X[:, 0], V[:, 0], 0.0, Lx)
        elif bcs == "periodic":
            _periodic_1d(X[:, 0], 0.0, Lx)
        else:
            raise ValueError(f"Unknown bcs '{bcs}'")
    else:
        xmin, xmax = self.info["xmin"], self.info["xmax"]
        ymin, ymax = self.info["ymin"], self.info["ymax"]
        if bcs == "reflective":
            _reflect_1d(X[:, 0], V[:, 0], xmin, xmax)
            _reflect_1d(X[:, 1], V[:, 1], ymin, ymax)
        elif bcs == "periodic":
            _periodic_1d(X[:, 0], xmin, xmax)
            _periodic_1d(X[:, 1], ymin, ymax)
        else:
            raise ValueError(f"Unknown bcs '{bcs}'")


def transport_step(self, dt):
    """Combined spatial advection + orientation drift + BCs + migration."""
    if self.nlocal == 0:
        # Still call migrate so all ranks participate.
        self.swarm.migrate(remove_sent_points=True)
        self.nlocal = self.swarm.getLocalSize()
        return

    self.swarm.sortGetAccess()
    celldm = self.swarm.getCellDMActive()
    coord_names = celldm.getCoordinateFields()
    pos = self.swarm.getField(coord_names[0])
    vel = self.swarm.getField("velocity")
    angle = self.swarm.getField("orientation")
    omega = self.swarm.getField("angular_velocity")

    X = pos.reshape(self.nlocal, self.mesh_dim)
    V = vel.reshape(self.nlocal, self.dim)

    # Spatial advection along the spatial dimensions (V[0:spatial_dim]).
    for d in range(self.spatial_dim):
        X[:, d] += V[:, d] * dt

    # Orientation drift; wrap to (0, 2π).
    angle[:, 0] += omega[:, 0] * dt
    np.mod(angle[:, 0], 2 * np.pi, out=angle[:, 0])

    _apply_bcs(self, X, V)

    self.swarm.restoreField(coord_names[0])
    self.swarm.restoreField("velocity")
    self.swarm.restoreField("orientation")
    self.swarm.restoreField("angular_velocity")
    self.swarm.sortRestoreAccess()
    self.swarm.migrate(remove_sent_points=True)
    self.nlocal = self.swarm.getLocalSize()


def _per_particle_density(self, theta_arr):
    """Build a per-particle orientation-density array of shape (nlocal, bins_theta).

    Each particle row is a copy of its owning cell's normalised θ histogram.
    Used by ``vlasov_kick_step`` to feed user-supplied force callables.
    """
    bins_theta = self.bins_theta
    edges_theta = np.linspace(0.0, 2.0 * np.pi, bins_theta + 1)
    cells = get_particle_cells(self)
    cell_lists = build_cell_lists(cells)

    density = np.zeros((self.nlocal, bins_theta), dtype=np.float64)
    for cell_id, idxs in cell_lists.items():
        if idxs.size == 0:
            continue
        hist, _ = np.histogram(theta_arr[idxs], bins=edges_theta)
        norm = hist.sum()
        if norm > 0:
            hist = hist / float(norm)
        density[idxs, :] = hist
    return density


def vlasov_kick_step(self, dt):
    """Apply translational and rotational mean-field kicks.

    V += G(θ, x, ρ_cell) · dt   if self.translational_force is set
    ω += F(θ, x, ρ_cell) · dt   if self.vlasov_force is set
    """
    if self.vlasov_force is None and self.translational_force is None:
        return
    if self.nlocal == 0:
        return

    self.swarm.sortGetAccess()
    celldm = self.swarm.getCellDMActive()
    coord_names = celldm.getCoordinateFields()
    pos = self.swarm.getField(coord_names[0])
    vel = self.swarm.getField("velocity")
    angle = self.swarm.getField("orientation")
    omega = self.swarm.getField("angular_velocity")

    X = pos.reshape(self.nlocal, self.mesh_dim)
    V = vel.reshape(self.nlocal, self.dim)
    theta_arr = angle.reshape(self.nlocal)

    density = _per_particle_density(self, theta_arr)

    if self.translational_force is not None:
        G = self.translational_force(angle, X, density)
        V[:, :] += G * dt

    if self.vlasov_force is not None:
        F = self.vlasov_force(angle, X, density)
        omega[:, 0] += F[:, 0] * dt

    self.swarm.restoreField(coord_names[0])
    self.swarm.restoreField("velocity")
    self.swarm.restoreField("orientation")
    self.swarm.restoreField("angular_velocity")
    self.swarm.sortRestoreAccess()
