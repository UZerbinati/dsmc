import numpy as np


def _reflect_1d(x, v, xmin, xmax):
    """Elastic reflection at domain boundaries [xmin, xmax]."""
    left = x < xmin
    right = x > xmax
    x[left] = 2.0 * xmin - x[left]
    v[left] = -v[left]
    x[right] = 2.0 * xmax - x[right]
    v[right] = -v[right]


def _reflect_cylinder(X, V, cx, cy, R):
    """Specular (elastic) reflection off a circular cylinder at (cx, cy) of radius R."""
    dx = X[:, 0] - cx
    dy = X[:, 1] - cy
    r2 = dx ** 2 + dy ** 2
    inside = r2 < R ** 2
    if not np.any(inside):
        return
    r = np.sqrt(r2[inside])
    # Outward unit normal at the particle's position
    nx = dx[inside] / r
    ny = dy[inside] / r
    # Reflect velocity: v' = v - 2(v·n)n
    v_dot_n = V[inside, 0] * nx + V[inside, 1] * ny
    V[inside, 0] -= 2.0 * v_dot_n * nx
    V[inside, 1] -= 2.0 * v_dot_n * ny
    # Place particle on cylinder surface
    X[inside, 0] = cx + R * nx
    X[inside, 1] = cy + R * ny


def _apply_cylinder_flow_bc(X, V, info):
    """
    Boundary conditions for 2D flow past a cylinder:
      - Periodic in x
      - Specular reflection on top/bottom walls (y = ymin, ymax)
      - Specular reflection on the cylinder surface
    """
    xmin = info["xmin"]
    xmax = info["xmax"]
    ymin = info["ymin"]
    ymax = info["ymax"]
    cx = info.get("cylinder_center_x", 0.0)
    cy = info.get("cylinder_center_y", 0.0)
    R  = info.get("cylinder_radius",   1.0)

    # Periodic wrap in x
    Lx = xmax - xmin
    X[:, 0] = xmin + (X[:, 0] - xmin) % Lx

    # Reflective top/bottom walls
    _reflect_1d(X[:, 1], V[:, 1], ymin, ymax)

    # Specular reflection off cylinder
    _reflect_cylinder(X, V, cx, cy, R)


def transport_step(self, dt):
    """Advance all particles by free streaming for time ``dt``.

    Positions are updated as X += V * dt along the effective spatial
    dimensions (1-D for ``sod``, 2-D for ``cylinder_flow``).  Boundary
    conditions are then applied (reflective walls / periodic wrap / cylinder
    specular reflection) before particles are migrated to their new owning
    ranks via ``DMSwarm.migrate``.

    Parameters
    ----------
    dt : float
        Time interval for the free-streaming substep.
    """
    self.swarm.sortGetAccess()
    celldm = self.swarm.getCellDMActive()
    coord_names = celldm.getCoordinateFields()
    pos = self.swarm.getField(coord_names[0])
    vel = self.swarm.getField("velocity")

    X = pos.reshape(self.nlocal, self.mesh_dim)
    V = vel.reshape(self.nlocal, self.dim)

    for d in range(self.effective_dim):
        X[:, d] += V[:, d] * dt

    if self.test == "sod":
        _reflect_1d(X[:, 0], V[:, 0], 0.0, self.info["Lx"])
    elif self.test == "cylinder_flow":
        _apply_cylinder_flow_bc(X, V, self.info)

    self.swarm.restoreField(coord_names[0])
    self.swarm.restoreField("velocity")
    self.swarm.sortRestoreAccess()
    self.swarm.migrate(remove_sent_points=True)
    self.nlocal = self.swarm.getLocalSize()
