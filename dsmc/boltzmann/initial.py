import numpy as np


def initialize_particles(self):
    if self.test == "sod":
        _initialize_sod(self)
        self.xlim = 8.0
        self.ylim = 8.0
    elif self.test == "cylinder_flow":
        _initialize_cylinder_flow(self)
    else:
        raise RuntimeError(f"[!] Unknown test: {self.test}")


def _initialize_sod(self):
    """
    Sod shock tube initial condition.

    Left state  (x < Lx/2): rho=1,     T=1.0
    Right state (x > Lx/2): rho=0.125, T=0.8

    The density ratio 8:1 is realised by particle count per cell.
    Each rank generates particles directly in its own spatial domain
    (determined from the DMDA decomposition), so no cross-rank migration
    is needed during initialisation.
    """
    Lx = self.info["Lx"]
    self.info["norm_rho"] = 0.5 * (1.0 + 0.125)

    # --- Determine this rank's x-domain from the DMDA ---
    # getRanges() returns ((xs, xe), (ys, ye)) of mesh-point indices.
    # Cell j spans [edges_x[j], edges_x[j+1]]; it is owned by the rank whose
    # range [xs, xe) contains j (the left mesh-point index).
    # The rank's physical domain is therefore [edges_x[xs-1], edges_x[xe-1]]
    # (with the convention that xs==0 anchors to 0).
    xs, xe = self.dm.getRanges()[0]
    x_start = self.edges_x[xs - 1] if xs > 0 else 0.0
    x_end   = self.edges_x[xe - 1]

    # --- Compute particle count for this rank proportional to Sod density ---
    # Sod density: 8 on [0, Lx/2], 1 on [Lx/2, Lx]
    mid = 0.5 * Lx
    total_weight = 8.0 * mid + 1.0 * (Lx - mid)          # = 4.5 for Lx=1
    left_in_domain  = max(0.0, min(x_end, mid) - x_start)
    right_in_domain = max(0.0, x_end - max(x_start, mid))
    local_weight = 8.0 * left_in_domain + 1.0 * right_in_domain

    n_local = max(1, int(round(self.N * local_weight / total_weight)))
    n_left  = int(round(8.0 * left_in_domain / total_weight * self.N))
    n_right = n_local - n_left

    # Resize swarm to hold n_local particles; keep a generous buffer for
    # later transport-step migrations.
    self.swarm.setLocalSizes(n_local, self.N)
    self.nlocal = n_local

    # --- Place particles in [x_start, x_end] ---
    X = np.zeros((n_local, self.mesh_dim))
    if n_left > 0:
        X[:n_left, 0] = self.rng.uniform(x_start, min(x_end, mid), n_left)
    if n_right > 0:
        X[n_left:, 0] = self.rng.uniform(max(x_start, mid), x_end, n_right)
    X[:, 1] = 0.5

    self.swarm.setPointCoordinates(X)
    self.nlocal = self.swarm.getLocalSize()

    # --- Assign Maxwellian velocities, smoothed to zero mean ---
    celldm = self.swarm.getCellDMActive()
    coord_names = celldm.getCoordinateFields()
    pos = self.swarm.getField(coord_names[0])
    vel = self.swarm.getField("velocity")
    wgt = self.swarm.getField("weight")

    X_local = pos.reshape(self.nlocal, self.mesh_dim)
    is_left = X_local[:, 0] < mid

    sigma_left  = np.sqrt(0.5 * self.temperature / self.mass)
    sigma_right = np.sqrt(0.5 * 0.8 * self.temperature / self.mass)

    gaussian = self.rng.normal(size=(self.nlocal, self.dim))
    for mask, sigma in [(is_left, sigma_left), (~is_left, sigma_right)]:
        if mask.sum() < 2:
            continue
        g = gaussian[mask]
        # Smooth twice: zero mean, unit std
        for _ in range(2):
            g = (g - g.mean(axis=0)) / np.where(g.std(axis=0) > 0, g.std(axis=0), 1.0)
        gaussian[mask] = g * sigma

    vel[:] = gaussian
    wgt[:] = 1.0

    self.swarm.restoreField(coord_names[0])
    self.swarm.restoreField("velocity")
    self.swarm.restoreField("weight")


def _initialize_cylinder_flow(self):
    """
    Flow past a cylinder initial condition.

    Particles are placed uniformly in [xmin, xmax] x [ymin, ymax] excluding
    the cylinder interior, using rejection sampling.  Velocities are sampled
    from a Maxwellian with mean drift u_inf in the x-direction.

    Boundary conditions (applied during transport):
      - Periodic in x
      - Specular (elastic) reflection on the cylinder surface
      - Specular (elastic) reflection on the top/bottom walls
    """
    xmin = self.info["xmin"]
    xmax = self.info["xmax"]
    ymin = self.info["ymin"]
    ymax = self.info["ymax"]
    cx   = self.info.get("cylinder_center_x", 0.0)
    cy   = self.info.get("cylinder_center_y", 0.0)
    R    = self.info.get("cylinder_radius",   1.0)
    u_inf = self.info.get("inflow_velocity",  1.0)

    # --- Determine this rank's spatial subdomain from the DMDA ---
    # getRanges() returns mesh-point index ranges ((xs, xe), (ys, ye)).
    # Convention (same as sod): rank owns physical x in
    #   [edges_x[xs-1], edges_x[xe-1]]  (xs==0 anchors to xmin)
    (xs, xe), (ys, ye) = self.dm.getRanges()
    nx = len(self.edges_x) - 1
    ny = len(self.edges_y) - 1

    x_lo = self.edges_x[xs - 1] if xs > 0 else self.edges_x[0]
    x_hi = self.edges_x[xe - 1]
    y_lo = self.edges_y[ys - 1] if ys > 0 else self.edges_y[0]
    y_hi = self.edges_y[ye - 1]

    # --- Particle count proportional to rank area ---
    total_area = (xmax - xmin) * (ymax - ymin)
    rank_area  = (x_hi - x_lo) * (y_hi - y_lo)
    n_local = max(1, int(round(self.N * rank_area / total_area)))

    self.swarm.setLocalSizes(n_local, self.N)
    self.nlocal = n_local

    # --- Rejection-sample positions outside the cylinder ---
    X = np.zeros((n_local, self.mesh_dim))
    count = 0
    batch = max(4 * n_local, 1000)
    while count < n_local:
        x_trial = self.rng.uniform(x_lo, x_hi, batch)
        y_trial = self.rng.uniform(y_lo, y_hi, batch)
        outside = (x_trial - cx) ** 2 + (y_trial - cy) ** 2 >= R ** 2
        valid_x = x_trial[outside]
        valid_y = y_trial[outside]
        take = min(len(valid_x), n_local - count)
        X[count:count + take, 0] = valid_x[:take]
        X[count:count + take, 1] = valid_y[:take]
        count += take

    self.swarm.setPointCoordinates(X)
    self.nlocal = self.swarm.getLocalSize()

    # --- Assign Maxwellian velocities with drift u_inf in x ---
    celldm = self.swarm.getCellDMActive()
    coord_names = celldm.getCoordinateFields()
    pos = self.swarm.getField(coord_names[0])
    vel = self.swarm.getField("velocity")
    wgt = self.swarm.getField("weight")

    n = self.nlocal
    sigma = np.sqrt(self.temperature / self.mass)

    gaussian = self.rng.normal(size=(n, self.dim))
    # Smooth to zero mean and unit variance before scaling
    for _ in range(2):
        g_mean = gaussian.mean(axis=0)
        g_std  = gaussian.std(axis=0)
        gaussian = (gaussian - g_mean) / np.where(g_std > 0, g_std, 1.0)

    V = vel.reshape(n, self.dim)
    V[:, 0] = gaussian[:, 0] * sigma + u_inf
    V[:, 1] = gaussian[:, 1] * sigma

    wgt[:] = 1.0

    self.swarm.restoreField(coord_names[0])
    self.swarm.restoreField("velocity")
    self.swarm.restoreField("weight")

    # Velocity-space histogram limits: inflow speed + several thermal widths
    v_max = u_inf + 4.0 * sigma
    self.xlim = max(v_max, 4.0)
    self.ylim = max(4.0 * sigma, 4.0)
