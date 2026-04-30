"""Initial conditions for the non-space-homogeneous CFMZ solver."""
import numpy as np


def initialize_particles(self):
    """Dispatch on ``self.test``; populate position, velocity, θ, ω."""
    if self.test == "sod_rod":
        _initialize_sod_rod(self)
    elif self.test == "uniform_1d":
        _initialize_uniform(self, dim=1)
    elif self.test == "uniform_2d":
        _initialize_uniform(self, dim=2)
    else:
        raise RuntimeError(f"[!] Unknown test: {self.test}")

    # Histogram-axis defaults shared with the homogeneous diagnostics plots.
    self.xlim = 8.0
    self.ylim = 8.0
    self.angular_min = 0.0
    self.angular_max = 2 * np.pi
    self.omega_min = -8.0
    self.omega_max = 8.0


def _smooth_twice(g, sigma):
    """Centre and rescale ``g`` so that columns have zero mean and std ``sigma``."""
    if g.shape[0] < 2:
        return g
    for _ in range(2):
        g = (g - g.mean(axis=0)) / np.where(g.std(axis=0) > 0, g.std(axis=0), 1.0)
    return g * sigma


def _initialize_sod_rod(self):
    """Sod-like 1-D rod shock tube.

    Left  (x < Lx/2): ρ_L = 1,     T_L = 1.0, θ uniform on [0, 2π).
    Right (x > Lx/2): ρ_R = 0.125, T_R = 0.8, θ ~ vonMises(π/2, κ).
    """
    Lx = self.info["Lx"]
    rho_L = self.info.get("rho_left", 1.0)
    rho_R = self.info.get("rho_right", 0.125)
    T_L = self.info.get("T_left", 1.0)
    T_R = self.info.get("T_right", 0.8)
    kappa = self.info.get("right_concentration", 2.0)
    self.info["norm_rho"] = 0.5 * (rho_L + rho_R)

    xs, xe = self.dm.getRanges()[0]
    x_start = self.edges_x[xs - 1] if xs > 0 else 0.0
    x_end = self.edges_x[xe - 1]

    mid = 0.5 * Lx
    total_weight = rho_L * mid + rho_R * (Lx - mid)
    left_in_domain = max(0.0, min(x_end, mid) - x_start)
    right_in_domain = max(0.0, x_end - max(x_start, mid))
    local_weight = rho_L * left_in_domain + rho_R * right_in_domain

    n_local = max(1, int(round(self.N * local_weight / total_weight)))
    n_left = int(round(rho_L * left_in_domain / total_weight * self.N))
    n_left = max(0, min(n_local, n_left))
    n_right = n_local - n_left

    self.swarm.setLocalSizes(n_local, self.N)
    self.nlocal = n_local

    X = np.zeros((n_local, self.mesh_dim))
    if n_left > 0:
        X[:n_left, 0] = self.rng.uniform(x_start, min(x_end, mid), n_left)
    if n_right > 0:
        X[n_left:, 0] = self.rng.uniform(max(x_start, mid), x_end, n_right)
    if self.mesh_dim >= 2:
        X[:, 1] = 0.5

    self.swarm.setPointCoordinates(X)
    self.nlocal = self.swarm.getLocalSize()

    celldm = self.swarm.getCellDMActive()
    coord_names = celldm.getCoordinateFields()
    pos = self.swarm.getField(coord_names[0])
    vel = self.swarm.getField("velocity")
    angle = self.swarm.getField("orientation")
    omega = self.swarm.getField("angular_velocity")
    wgt = self.swarm.getField("weight")

    n = self.nlocal
    if n > 0:
        X_local = pos.reshape(n, self.mesh_dim)
        is_left = X_local[:, 0] < mid
        is_right = ~is_left

        m_mass = self.info["mass"]
        I_inertia = self.info["inertia"]
        sigma_v_L = np.sqrt(T_L / m_mass)
        sigma_v_R = np.sqrt(T_R / m_mass)
        sigma_w_L = np.sqrt(T_L / I_inertia)
        sigma_w_R = np.sqrt(T_R / I_inertia)

        gauss_v = self.rng.normal(size=(n, self.dim))
        gauss_w = self.rng.normal(size=(n, 1))
        for mask, sv, sw in [(is_left, sigma_v_L, sigma_w_L),
                             (is_right, sigma_v_R, sigma_w_R)]:
            count = int(mask.sum())
            if count >= 2:
                gauss_v[mask] = _smooth_twice(gauss_v[mask], sv)
                gauss_w[mask] = _smooth_twice(gauss_w[mask], sw)

        vel[:] = gauss_v
        omega[:] = gauss_w

        # Orientations: uniform on left, von Mises at π/2 on right.
        angle_arr = np.empty((n, 1))
        if is_left.any():
            angle_arr[is_left, 0] = self.rng.uniform(0.0, 2 * np.pi, int(is_left.sum()))
        if is_right.any():
            samples = self.rng.vonmises(0.0, kappa, int(is_right.sum())) + 0.5 * np.pi
            angle_arr[is_right, 0] = np.mod(samples, 2 * np.pi)
        # Avoid hitting the (0, 2π) boundary that diagnostics() guards against.
        angle_arr = np.clip(angle_arr, 1e-12, 2 * np.pi - 1e-12)
        angle[:] = angle_arr
        wgt[:] = 1.0

    self.swarm.restoreField(coord_names[0])
    self.swarm.restoreField("velocity")
    self.swarm.restoreField("orientation")
    self.swarm.restoreField("angular_velocity")
    self.swarm.restoreField("weight")


def _initialize_uniform(self, dim):
    """Uniform IC: spatial density uniform on the full domain, θ ~ Uniform(0, 2π)."""
    if dim == 1:
        Lx = self.info["Lx"]
        xs, xe = self.dm.getRanges()[0]
        x_lo = self.edges_x[xs - 1] if xs > 0 else 0.0
        x_hi = self.edges_x[xe - 1]
        rank_extent = x_hi - x_lo
        total_extent = Lx
    else:
        (xs, xe), (ys, ye) = self.dm.getRanges()
        x_lo = self.edges_x[xs - 1] if xs > 0 else self.edges_x[0]
        x_hi = self.edges_x[xe - 1]
        y_lo = self.edges_y[ys - 1] if ys > 0 else self.edges_y[0]
        y_hi = self.edges_y[ye - 1]
        rank_extent = (x_hi - x_lo) * (y_hi - y_lo)
        total_extent = (self.info["xmax"] - self.info["xmin"]) * (self.info["ymax"] - self.info["ymin"])

    self.info["norm_rho"] = 1.0
    n_local = max(1, int(round(self.N * rank_extent / total_extent)))
    self.swarm.setLocalSizes(n_local, self.N)
    self.nlocal = n_local

    X = np.zeros((n_local, self.mesh_dim))
    X[:, 0] = self.rng.uniform(x_lo, x_hi, n_local)
    if dim == 2 and self.mesh_dim >= 2:
        X[:, 1] = self.rng.uniform(y_lo, y_hi, n_local)
    elif self.mesh_dim >= 2:
        X[:, 1] = 0.5
    self.swarm.setPointCoordinates(X)
    self.nlocal = self.swarm.getLocalSize()

    n = self.nlocal
    vel = self.swarm.getField("velocity")
    angle = self.swarm.getField("orientation")
    omega = self.swarm.getField("angular_velocity")
    wgt = self.swarm.getField("weight")

    if n > 0:
        if self.T_bath is not None and self.init_at_T_bath:
            m_mass = self.info["mass"]
            I_inertia = self.info["inertia"]
            vel[:] = self.rng.normal(0.0, np.sqrt(self.T_bath / m_mass), (n, self.dim))
            omega[:] = self.rng.normal(0.0, np.sqrt(self.T_bath / I_inertia), (n, 1))
        else:
            vel[:] = self.rng.uniform(-0.5, 0.5, (n, self.dim))
            omega[:] = self.rng.uniform(-0.25, 0.25, (n, 1))
        ang = self.rng.uniform(0.0, 2 * np.pi, (n, 1))
        ang = np.clip(ang, 1e-12, 2 * np.pi - 1e-12)
        angle[:] = ang
        wgt[:] = 1.0

    self.swarm.restoreField("velocity")
    self.swarm.restoreField("orientation")
    self.swarm.restoreField("angular_velocity")
    self.swarm.restoreField("weight")
