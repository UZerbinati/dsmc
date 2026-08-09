import numpy as np
from mpi4py import MPI


def _replenish_open_inlet(self, coord_names):
    """Restore the global particle count after ``migrate`` by injecting the
    deficit at the inlet reservoir (on rank 0).  ``migrate`` silently drops a
    small fraction of particles each step (cell-location / MPI-transfer edge
    cases); left uncorrected this compounds to a ~50% loss over a long run.
    Replenishing to ``self.N`` keeps the statistics stationary."""
    cur = self.comm.allreduce(self.swarm.getLocalSize(), op=MPI.SUM)
    deficit = int(self.N - cur)
    # Only pay the (re-migrate) cost once >1% has leaked: keeps N within 1% of
    # target with negligible per-step overhead (vital at 10^8 particles).
    if deficit < 0.01 * self.N:
        return
    if self.rank == 0:
        info = self.info
        xmin, xmax = info["xmin"], info["xmax"]
        ymin, ymax = info["ymin"], info["ymax"]
        u_inf = info.get("inflow_velocity", 1.0)
        sigma = np.sqrt(self.temperature / self.mass)
        band = 0.02 * (xmax - xmin)
        old = self.swarm.getLocalSize()
        new = old + deficit
        self.swarm.setLocalSizes(new, 0)
        p = self.swarm.getField(coord_names[0]).reshape(new, self.mesh_dim)
        vv = self.swarm.getField("velocity").reshape(new, self.dim)
        ww = self.swarm.getField("weight")
        p[old:, 0] = xmin + self.rng.uniform(0.0, band, deficit)
        p[old:, 1] = self.rng.uniform(ymin, ymax, deficit)
        vv[old:, 0] = u_inf + sigma * self.rng.standard_normal(deficit)
        vv[old:, 1] = sigma * self.rng.standard_normal(deficit)
        ww[old:] = ww[0]
        self.swarm.restoreField(coord_names[0])
        self.swarm.restoreField("velocity")
        self.swarm.restoreField("weight")
    # re-migrate so the injected particles are assigned cells / owners
    self.swarm.migrate(remove_sent_points=True)


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
    Periodic-in-x boundary conditions for 2D flow past a cylinder (original):
      - Periodic in x
      - Specular reflection on top/bottom walls (y = ymin, ymax)
      - Specular reflection on the cylinder surface

    NOTE: the periodic wrap feeds the downstream wake back onto the inlet,
    which suppresses a clean von Karman street.  Use ``bc_type="open"`` for
    genuine open-flow boundaries (``_apply_cylinder_flow_open_bc``).
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


def _apply_cylinder_flow_open_bc(X, V, info, rng, sigma):
    """
    Open (inflow/outflow) boundary conditions for 2D flow past a cylinder --
    the configuration that supports a clean von Karman vortex street:

      - Specular reflection off the cylinder surface.
      - Free-slip specular top/bottom walls (y = ymin, ymax).
      - Reservoir INFLOW at x = xmin and absorbing OUTFLOW at x = xmax,
        realised by *particle recycling*: every particle that leaves the
        domain through xmax (advected downstream in the wake) -- or through
        xmin -- is re-injected in a thin slab at the inlet with a fresh
        drifting-Maxwellian velocity (mean u_inf in x, thermal width sigma)
        and a uniformly random y.

    Recycling keeps the particle count fixed (no DMSwarm add/remove), pins the
    inlet to the equilibrium reservoir so the mean flow is sustained, and lets
    the wake convect out of the domain instead of recirculating through a
    periodic wrap.  The wake's velocity memory is erased on re-injection, so
    the inflow stays clean.
    """
    xmin = info["xmin"]
    xmax = info["xmax"]
    ymin = info["ymin"]
    ymax = info["ymax"]
    cx = info.get("cylinder_center_x", 0.0)
    cy = info.get("cylinder_center_y", 0.0)
    R  = info.get("cylinder_radius",   1.0)
    u_inf = info.get("inflow_velocity", 1.0)

    # Cylinder + free-slip y-walls first (a particle may both reflect and exit).
    _reflect_cylinder(X, V, cx, cy, R)
    _reflect_1d(X[:, 1], V[:, 1], ymin, ymax)

    # Recycle particles that have left through the x-boundaries: re-inject them
    # at the inlet from the drifting-Maxwellian reservoir.
    out = (X[:, 0] > xmax) | (X[:, 0] < xmin)
    m = int(np.count_nonzero(out))
    if m > 0:
        band = 0.02 * (xmax - xmin)             # inlet slab a couple of cells thick
        X[out, 0] = xmin + rng.uniform(0.0, band, m)
        X[out, 1] = rng.uniform(ymin, ymax, m)
        V[out, 0] = u_inf + sigma * rng.standard_normal(m)
        V[out, 1] = sigma * rng.standard_normal(m)

    # Safety net: clamp every particle strictly inside the box so the subsequent
    # DMSwarm.migrate never silently drops one -- fast double-crossers, cylinder
    # ejections and floating-point boundary cases were the ~50% particle leak.
    eps = 1e-9 * (xmax - xmin)
    X[:, 0] = np.clip(X[:, 0], xmin + eps, xmax - eps)
    X[:, 1] = np.clip(X[:, 1], ymin + eps, ymax - eps)


def transport_step(self, dt):
    """Advance all particles by free streaming for time ``dt``.

    Positions are updated as X += V * dt along the effective spatial
    dimensions (1-D for ``sod``, 2-D for ``cylinder_flow``).  Boundary
    conditions are then applied (reflective walls / periodic wrap or open
    inflow-outflow / cylinder specular reflection) before particles are
    migrated to their new owning ranks via ``DMSwarm.migrate``.

    For ``cylinder_flow`` the boundary treatment is selected by
    ``info["bc_type"]``: ``"periodic"`` (default, original) or ``"open"``
    (reservoir inflow / absorbing outflow via particle recycling).

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
        if self.info.get("bc_type", "periodic") == "open":
            sigma = np.sqrt(self.temperature / self.mass)
            _apply_cylinder_flow_open_bc(X, V, self.info, self.rng, sigma)
        else:
            _apply_cylinder_flow_bc(X, V, self.info)

    self.swarm.restoreField(coord_names[0])
    self.swarm.restoreField("velocity")
    self.swarm.sortRestoreAccess()
    self.swarm.migrate(remove_sent_points=True)
    if self.test == "cylinder_flow" and self.info.get("bc_type", "periodic") == "open":
        _replenish_open_inlet(self, coord_names)
    self.nlocal = self.swarm.getLocalSize()
