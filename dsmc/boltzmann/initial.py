import numpy as np


def initialize_particles(self):
    if self.test == "sod":
        _initialize_sod(self)
    else:
        raise RuntimeError(f"[!] Unknown test: {self.test}")

    self.xlim = 8.0
    self.ylim = 8.0


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
