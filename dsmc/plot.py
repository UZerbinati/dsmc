import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pickle
import re
from dsmc.utils import init_plot, pv_cmap, fig_axes


def plot_history(self, prefix=""):
    """Write time-history plots (temperature, energy, circular variance) to PDF/PNG.

    One figure per quantity is saved as ``<prefix>_<quantity>.pdf/png``.
    Only quantities present in ``self.history`` are plotted.

    The function additionally sweeps ``self.history`` for keys matching
    ``circular_var_n{n}`` and overlays them on a single
    ``<prefix>_variance_modes.pdf/png`` figure (one curve per harmonic
    n).  Useful for the discotic solver, which tracks R₁, R₂, R₄, R₆
    simultaneously.
    """
    time = np.array(self.history["step"]) * self.dt

    to_plot = []
    if "temperature" in self.history:
        to_plot.append((self.history["temperature"], r"$T$", "_temperature"))
    if "energy" in self.history:
        to_plot.append((self.history["energy"], r"$E_{\mathrm{kin}}$", "_energy"))
    if "interaction_energy" in self.history:
        to_plot.append((self.history["interaction_energy"], r"$\mathcal{E}[\rho]$", "_interaction_energy"))
    if "total_energy" in self.history and "total_energy_rot" in self.history:
        fig, ax, _ = fig_axes()
        time = np.array(self.history["step"]) * self.dt
        ax.plot(time, np.array(self.history["total_energy"]),
                color="black", linewidth=1.5, label=r"$E_{\mathrm{kin}} + \mathcal{E}[\rho]$")
        ax.plot(time, np.array(self.history["total_energy_rot"]),
                color="red", linewidth=1.5, linestyle="--",
                label=r"$E_{\mathrm{kin,rot}} + \mathcal{E}[\rho]$")
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"Energy")
        ax.legend()
        ax.tick_params(which="both", direction="in", top=True, right=True)
        fig.savefig(f"{prefix}_total_energy.pdf")
        fig.savefig(f"{prefix}_total_energy.png", dpi=400)
        plt.close(fig)
    elif "total_energy" in self.history:
        to_plot.append((self.history["total_energy"], r"$E_{\mathrm{kin}} + \,\mathcal{E}[\rho]$", "_total_energy"))
    if "circular_var" in self.history:
        to_plot.append((self.history["circular_var"], r"$\mathrm{Var}(\theta)$", "_variance"))

    for data, ylabel, fname_suffix in to_plot:
        fig, ax, _ = fig_axes()
        ax.plot(time, np.array(data), color="black", linewidth=1.5)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(ylabel)
        ax.tick_params(which="both", direction="in", top=True, right=True)
        fig.savefig(f"{prefix}{fname_suffix}.pdf")
        fig.savefig(f"{prefix}{fname_suffix}.png", dpi=400)
        plt.close(fig)

    # Multi-harmonic order parameters: R_n curves on a single axis.
    mode_re = re.compile(r"^circular_var_n(\d+)$")
    mode_keys = sorted(
        ((int(m.group(1)), key) for key in self.history.keys() if (m := mode_re.match(key))),
        key=lambda t: t[0],
    )
    if mode_keys:
        fig, ax, _ = fig_axes()
        for n, key in mode_keys:
            ax.plot(time, 1.0 - np.array(self.history[key]),
                    linewidth=1.5, label=fr"$R_{{{n}}}$")
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$R_n = |\langle e^{i n \theta}\rangle|$")
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.tick_params(which="both", direction="in", top=True, right=True)
        fig.savefig(f"{prefix}_variance_modes.pdf")
        fig.savefig(f"{prefix}_variance_modes.png", dpi=400)
        plt.close(fig)


# ---------------------------------------------------------------------------
# CFMZ-specific
# ---------------------------------------------------------------------------

def plot_histograms(self, prefix=""):
    """Write velocity and angular distribution plots to PDF/PNG (CFMZ needle solver).

    All particle data are gathered to rank 0.  Saves:
      - 2D velocity histogram     ``<prefix>_vel.pdf/png`` (+ pickle)
      - vx marginal               ``<prefix>_vel_x.pdf/png``
      - vy marginal               ``<prefix>_vel_y.pdf/png``
      - 2D (θ, ω) histogram       ``<prefix>_angular.pdf/png`` (+ pickle)
      - θ marginal                ``<prefix>_theta.pdf/png``
      - ω marginal                ``<prefix>_omega.pdf/png``

    Marginals are overlaid with the corresponding Maxwellian.
    Only rank 0 writes files; all other ranks return immediately after
    gathering.
    """
    step = len(self.history["temperature"]) - 1
    if self.vlasov_force:
        Maxwellian = self.maxwellian(step)
    else:
        Maxwellian = self.maxwellian(0)

    from mpi4py import MPI as _MPI

    vel = self.swarm.getField("velocity")
    Vlocal = vel.reshape(self.nlocal, self.dim).copy()
    self.swarm.restoreField("velocity")

    angle = self.swarm.getField("orientation")
    Alocal = angle.reshape(self.nlocal, 1).copy()
    self.swarm.restoreField("orientation")

    omega = self.swarm.getField("angular_velocity")
    Wlocal = omega.reshape(self.nlocal, 1).copy()
    self.swarm.restoreField("angular_velocity")

    # Each rank builds its local 2D histograms and reduces to rank 0.
    # This transfers O(bins²) instead of O(N) data over MPI.
    local_H_v, xedges, yedges = np.histogram2d(
        Vlocal[:, 0], Vlocal[:, 1], bins=(self.grid_x, self.grid_y),
    )
    local_H_v = np.ascontiguousarray(local_H_v)
    H_v = np.zeros_like(local_H_v)
    self.comm.Reduce(local_H_v, H_v, op=_MPI.SUM, root=0)

    local_H_ang, thetaedges, omegaedges = np.histogram2d(
        Alocal[:, 0], Wlocal[:, 0], bins=(self.grid_angular, self.grid_omega),
    )
    local_H_ang = np.ascontiguousarray(local_H_ang)
    H_ang = np.zeros_like(local_H_ang)
    self.comm.Reduce(local_H_ang, H_ang, op=_MPI.SUM, root=0)

    if self.rank != 0:
        return

    # Normalise
    H_v   = H_v   / (np.sum(H_v)   * self.delta_x       * self.delta_y      )
    H_ang = H_ang / (np.sum(H_ang) * self.delta_angular  * self.delta_omega  )

    # --- 2D velocity histogram ---
    fig, ax, cax = fig_axes(colorbar=True)
    pcm = ax.pcolormesh(xedges, yedges, H_v.T, cmap=pv_cmap, shading="auto", rasterized=True)
    ax.set_xlabel(r"$v_x$")
    ax.set_ylabel(r"$v_y$")
    ax.set_xlim(-self.xlim, self.xlim)
    ax.set_ylim(-self.ylim, self.ylim)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    cbar = fig.colorbar(pcm, cax=cax, format=mticker.ScalarFormatter(useMathText=True))
    cbar.formatter.set_scientific(True)
    cbar.formatter.set_powerlimits((-2, 2))
    cbar.update_ticks()
    #cbar.set_label(r"$f(v)$")
    fig.savefig(f"{prefix}_vel.pdf", bbox_inches="tight")
    fig.savefig(f"{prefix}_vel.png", dpi=400, bbox_inches="tight")
    plt.close(fig)
    if self.dump == "hist":
        with open(f"{prefix}_vel.pickle", "wb") as fp:
            pickle.dump({"hist": H_v, "xedges": xedges, "yedges": yedges}, fp)

    # --- vx marginal ---
    H_x = np.sum(H_v, axis=1) * self.delta_y
    H_x = H_x / (np.sum(H_x) * self.delta_x)
    fig, ax, _ = fig_axes()
    ax.plot(
        [x + 0.5 * self.delta_x for x in xedges[:-1]], H_x,
        linestyle="None", marker="o", markersize=6,
        markerfacecolor="none", markeredgecolor="red", markeredgewidth=1.0,
        label="DSMC",
    )
    ax.plot(xedges, Maxwellian[1], color="black", linewidth=1.5, label="Maxwellian")
    ax.set_xlabel(r"$v_x$")
    #ax.set_ylabel(r"$f(v_x)$")
    ax.legend()
    ax.tick_params(which="both", direction="in", top=True, right=True)
    fig.savefig(f"{prefix}_vel_x.pdf")
    fig.savefig(f"{prefix}_vel_x.png", dpi=400)
    plt.close(fig)

    # --- vy marginal ---
    H_y = np.sum(H_v, axis=0) * self.delta_x
    H_y = H_y / (np.sum(H_y) * self.delta_y)
    fig, ax, _ = fig_axes()
    ax.plot(
        [y + 0.5 * self.delta_y for y in yedges[:-1]], H_y,
        linestyle="None", marker="o", markersize=6,
        markerfacecolor="none", markeredgecolor="red", markeredgewidth=1.0,
        label="DSMC",
    )
    ax.plot(yedges, Maxwellian[2], color="black", linewidth=1.5, label="Maxwellian")
    ax.set_xlabel(r"$v_y$")
    #ax.set_ylabel(r"$f(v_y)$")
    ax.legend()
    ax.tick_params(which="both", direction="in", top=True, right=True)
    fig.savefig(f"{prefix}_vel_y.pdf")
    fig.savefig(f"{prefix}_vel_y.png", dpi=400)
    plt.close(fig)

    # --- 2D angular histogram ---
    fig, ax, cax = fig_axes(colorbar=True)
    pcm = ax.pcolormesh(thetaedges, omegaedges, H_ang.T, cmap=pv_cmap, shading="auto", rasterized=True)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\omega$")
    ax.set_xlim(self.angular_min, self.angular_max)
    ax.set_ylim(self.omega_min, self.omega_max)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    cbar = fig.colorbar(pcm, cax=cax, format=mticker.ScalarFormatter(useMathText=True))
    cbar.formatter.set_scientific(True)
    cbar.formatter.set_powerlimits((-2, 2))
    cbar.update_ticks()
    #cbar.set_label(r"$f(\theta,\omega)$")
    fig.savefig(f"{prefix}_angular.pdf", bbox_inches="tight")
    fig.savefig(f"{prefix}_angular.png", dpi=400, bbox_inches="tight")
    plt.close(fig)
    if self.dump == "hist":
        with open(f"{prefix}_angular.pickle", "wb") as fp:
            pickle.dump({"hist": H_ang, "theta_edges": thetaedges, "omega_edges": omegaedges}, fp)

    # --- theta marginal ---
    H_theta = np.sum(H_ang, axis=1) * self.delta_omega
    H_theta = H_theta / (np.sum(H_theta) * self.delta_angular)
    fig, ax, _ = fig_axes()
    ax.plot(
        [theta + 0.5 * self.delta_angular for theta in thetaedges[:-1]], H_theta,
        linestyle="None", marker="o", markersize=6,
        markerfacecolor="none", markeredgecolor="red", markeredgewidth=1.0,
        label="DSMC",
    )
    ax.set_xlabel(r"$\theta$")
    #ax.set_ylabel(r"$f(\theta)$")
    ax.legend()
    mv = np.mean(H_theta)
    ymin = min(H_theta.min(), mv - 0.1)
    ymax = max(H_theta.max(), mv + 0.1)
    margin = 0.02 * (ymax - ymin)
    ax.set_ylim(ymin - margin, ymax + margin)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    fig.savefig(f"{prefix}_theta.pdf")
    fig.savefig(f"{prefix}_theta.png", dpi=400)
    plt.close(fig)

    # --- omega marginal ---
    H_omega = np.sum(H_ang, axis=0) * self.delta_angular
    H_omega = H_omega / (np.sum(H_omega) * self.delta_omega)
    fig, ax, _ = fig_axes()
    ax.plot(
        [w + 0.5 * self.delta_omega for w in omegaedges[:-1]], H_omega,
        linestyle="None", marker="o", markersize=6,
        markerfacecolor="none", markeredgecolor="red", markeredgewidth=1.0,
        label="DSMC",
    )
    ax.plot(omegaedges, Maxwellian[3], color="black", linewidth=1.5, label="Maxwellian")
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$f(\omega)$")
    ax.legend()
    ax.tick_params(which="both", direction="in", top=True, right=True)
    fig.savefig(f"{prefix}_omega.pdf")
    fig.savefig(f"{prefix}_omega.png", dpi=400)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Boltzmann-specific
# ---------------------------------------------------------------------------

def plot_observables(self, prefix=""):
    """Spatial profiles of density, mean velocity, and temperature."""
    from mpi4py import MPI

    vel = self.swarm.getField("velocity")
    V = vel.reshape(self.nlocal, self.dim).copy()
    self.swarm.restoreField("velocity")

    celldm = self.swarm.getCellDMActive()
    coord_names = celldm.getCoordinateFields()
    pos = self.swarm.getField(coord_names[0])
    X = pos.reshape(self.nlocal, self.mesh_dim).copy()
    self.swarm.restoreField(coord_names[0])

    xpos = X[:, 0]
    edges = self.edges_x
    dx = edges[1] - edges[0]

    local_counts = np.histogram(xpos, bins=edges)[0].astype(float)
    local_vel_sum = np.histogram(xpos, bins=edges, weights=V[:, 0])[0]
    local_ke_sum = np.histogram(xpos, bins=edges, weights=V[:, 0] ** 2 + V[:, 1] ** 2)[0]

    global_counts = self.comm.allreduce(local_counts, op=MPI.SUM)
    global_vel_sum = self.comm.allreduce(local_vel_sum, op=MPI.SUM)
    global_ke_sum = self.comm.allreduce(local_ke_sum, op=MPI.SUM)

    if self.rank != 0:
        return

    safe = np.where(global_counts > 0, global_counts, 1.0)
    rho_x = (global_counts / self.N) * (self.bins / self.info["Lx"]) * self.info["norm_rho"]
    vel_x = global_vel_sum / safe
    temp_x = self.mass * global_ke_sum / safe - self.mass * vel_x ** 2

    x_centers = 0.5 * (edges[:-1] + edges[1:])

    for data, ylabel, fname in [
        (rho_x, r"$\rho$", f"{prefix}_density.pdf"),
        (vel_x, r"$u_x$", f"{prefix}_velocity.pdf"),
        (temp_x, r"$T$", f"{prefix}_temperature.pdf"),
    ]:
        fig, ax, _ = fig_axes()
        ax.plot(x_centers, data, color="black")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(ylabel)
        ax.tick_params(which="both", direction="in", top=True, right=True)
        fig.savefig(fname)
        fig.savefig(fname.replace(".pdf", ".png"), dpi=400)
        plt.close(fig)

    with open(f"{prefix}_observables.pickle", "wb") as fp:
        pickle.dump({"x": x_centers, "rho": rho_x, "vel_x": vel_x, "temp": temp_x}, fp)


def plot_cylinder_flow_observables(self, prefix=""):
    """2D spatial fields (density, speed, temperature) for flow past a cylinder."""
    from mpi4py import MPI

    vel = self.swarm.getField("velocity")
    V = vel.reshape(self.nlocal, self.dim).copy()
    self.swarm.restoreField("velocity")

    celldm = self.swarm.getCellDMActive()
    coord_names = celldm.getCoordinateFields()
    pos = self.swarm.getField(coord_names[0])
    X = pos.reshape(self.nlocal, self.mesh_dim).copy()
    self.swarm.restoreField(coord_names[0])

    edges_x = self.edges_x
    edges_y = self.edges_y
    xpos = X[:, 0]
    ypos = X[:, 1]

    # Per-rank 2D histograms
    local_counts, _, _ = np.histogram2d(xpos, ypos, bins=(edges_x, edges_y))
    local_vx, _, _ = np.histogram2d(xpos, ypos, bins=(edges_x, edges_y), weights=V[:, 0])
    local_vy, _, _ = np.histogram2d(xpos, ypos, bins=(edges_x, edges_y), weights=V[:, 1])
    local_ke, _, _ = np.histogram2d(
        xpos, ypos, bins=(edges_x, edges_y), weights=V[:, 0] ** 2 + V[:, 1] ** 2
    )

    global_counts = self.comm.allreduce(local_counts, op=MPI.SUM)
    global_vx     = self.comm.allreduce(local_vx,     op=MPI.SUM)
    global_vy     = self.comm.allreduce(local_vy,     op=MPI.SUM)
    global_ke     = self.comm.allreduce(local_ke,     op=MPI.SUM)

    if self.rank != 0:
        return

    safe = np.where(global_counts > 0, global_counts, 1.0)
    cell_area = (edges_x[1] - edges_x[0]) * (edges_y[1] - edges_y[0])
    total_n = global_counts.sum()
    rho   = global_counts / (max(total_n, 1) * cell_area)
    ux    = global_vx / safe
    uy    = global_vy / safe
    speed = np.sqrt(ux ** 2 + uy ** 2)
    temp  = self.mass * global_ke / safe - self.mass * (ux ** 2 + uy ** 2)
    temp  = np.where(global_counts > 1, temp, 0.0)

    # Cylinder outline for overlay
    cx = self.info.get("cylinder_center_x", 0.0)
    cy = self.info.get("cylinder_center_y", 0.0)
    R  = self.info.get("cylinder_radius",   1.0)
    theta_cyl = np.linspace(0.0, 2.0 * np.pi, 300)
    cyl_x = cx + R * np.cos(theta_cyl)
    cyl_y = cy + R * np.sin(theta_cyl)

    for data, label, fname_suffix in [
        (rho.T,   r"$\rho$",  "_density"),
        (speed.T, r"$|u|$",   "_speed"),
        (temp.T,  r"$T$",     "_temperature"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4))
        pcm = ax.pcolormesh(
            edges_x, edges_y, data,
            cmap=pv_cmap, shading="auto", rasterized=True,
        )
        ax.plot(cyl_x, cyl_y, color="white", linewidth=1.0)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_aspect("equal")
        ax.tick_params(which="both", direction="in", top=True, right=True)
        cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label(label, fontsize=10)
        fig.savefig(f"{prefix}{fname_suffix}.pdf")
        fig.savefig(f"{prefix}{fname_suffix}.png", dpi=400)
        plt.close(fig)

    import pickle
    with open(f"{prefix}_observables.pickle", "wb") as fp:
        pickle.dump({
            "x": 0.5 * (edges_x[:-1] + edges_x[1:]),
            "y": 0.5 * (edges_y[:-1] + edges_y[1:]),
            "rho": rho, "ux": ux, "uy": uy, "temp": temp,
        }, fp)


# ---------------------------------------------------------------------------
# CFMZ non-homogeneous spatial profiles
# ---------------------------------------------------------------------------

def plot_cfmz_observables(self, prefix=""):
    """Spatial profiles of ρ(x), u(x), T(x), nematic order S(x).

    Reduces per-cell counts and weighted moments via MPI allreduce, then on
    rank 0 saves PDF/PNG plots and a pickle of the profiles.  Falls back to a
    single panel per quantity for the 1-D case; for 2-D the same quantities
    are plotted as 2-D heat maps.
    """
    from mpi4py import MPI

    if self.nlocal == 0:
        # Still need to participate in the allreduce so collective ops sync.
        n_x = len(self.edges_x) - 1
        n_y = (len(self.edges_y) - 1) if self.spatial_dim == 2 else 1
        zero = np.zeros((n_x, n_y), dtype=np.float64) if self.spatial_dim == 2 else np.zeros(n_x)
        local_counts = zero.copy()
        local_vx = zero.copy()
        local_vy = zero.copy()
        local_ke = zero.copy()
        local_re = zero.copy()
        local_im = zero.copy()
    else:
        celldm = self.swarm.getCellDMActive()
        coord_names = celldm.getCoordinateFields()
        pos = self.swarm.getField(coord_names[0])
        vel = self.swarm.getField("velocity")
        angle = self.swarm.getField("orientation")

        X = pos.reshape(self.nlocal, self.mesh_dim).copy()
        V = vel.reshape(self.nlocal, self.dim).copy()
        theta = angle.reshape(self.nlocal).copy()

        self.swarm.restoreField(coord_names[0])
        self.swarm.restoreField("velocity")
        self.swarm.restoreField("orientation")

        # Nematic director uses 2θ so π-equivalent rod orientations align.
        z2 = np.exp(2j * theta)

        if self.spatial_dim == 1:
            xpos = X[:, 0]
            edges = self.edges_x
            local_counts = np.histogram(xpos, bins=edges)[0].astype(float)
            local_vx = np.histogram(xpos, bins=edges, weights=V[:, 0])[0]
            local_vy = np.histogram(xpos, bins=edges, weights=V[:, 1])[0]
            local_ke = np.histogram(xpos, bins=edges, weights=V[:, 0] ** 2 + V[:, 1] ** 2)[0]
            local_re = np.histogram(xpos, bins=edges, weights=z2.real)[0]
            local_im = np.histogram(xpos, bins=edges, weights=z2.imag)[0]
        else:
            xpos = X[:, 0]
            ypos = X[:, 1]
            ex, ey = self.edges_x, self.edges_y
            local_counts, _, _ = np.histogram2d(xpos, ypos, bins=(ex, ey))
            local_vx, _, _ = np.histogram2d(xpos, ypos, bins=(ex, ey), weights=V[:, 0])
            local_vy, _, _ = np.histogram2d(xpos, ypos, bins=(ex, ey), weights=V[:, 1])
            local_ke, _, _ = np.histogram2d(xpos, ypos, bins=(ex, ey), weights=V[:, 0] ** 2 + V[:, 1] ** 2)
            local_re, _, _ = np.histogram2d(xpos, ypos, bins=(ex, ey), weights=z2.real)
            local_im, _, _ = np.histogram2d(xpos, ypos, bins=(ex, ey), weights=z2.imag)

    g_counts = self.comm.allreduce(local_counts, op=MPI.SUM)
    g_vx = self.comm.allreduce(local_vx, op=MPI.SUM)
    g_vy = self.comm.allreduce(local_vy, op=MPI.SUM)
    g_ke = self.comm.allreduce(local_ke, op=MPI.SUM)
    g_re = self.comm.allreduce(local_re, op=MPI.SUM)
    g_im = self.comm.allreduce(local_im, op=MPI.SUM)

    if self.rank != 0:
        return

    safe = np.where(g_counts > 0, g_counts, 1.0)
    mass = self.info["mass"]

    if self.spatial_dim == 1:
        Lx = self.info["Lx"]
        rho_x = (g_counts / max(self.N, 1)) * (self.bins / Lx) * self.info["norm_rho"]
        vel_x = g_vx / safe
        temp_x = mass * g_ke / safe - mass * (vel_x ** 2 + (g_vy / safe) ** 2)
        S_x = np.sqrt((g_re / safe) ** 2 + (g_im / safe) ** 2)

        x_centers = 0.5 * (self.edges_x[:-1] + self.edges_x[1:])

        for data, ylabel, fname in [
            (rho_x, r"$\rho$", f"{prefix}_density.pdf"),
            (vel_x, r"$u_x$", f"{prefix}_velocity.pdf"),
            (temp_x, r"$T$", f"{prefix}_temperature.pdf"),
            (S_x, r"$S$", f"{prefix}_nematic.pdf"),
        ]:
            fig, ax, _ = fig_axes()
            ax.plot(x_centers, data, color="black")
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(ylabel)
            ax.tick_params(which="both", direction="in", top=True, right=True)
            fig.savefig(fname)
            fig.savefig(fname.replace(".pdf", ".png"), dpi=400)
            plt.close(fig)

        with open(f"{prefix}_observables.pickle", "wb") as fp:
            pickle.dump({"x": x_centers, "rho": rho_x, "vel_x": vel_x,
                         "temp": temp_x, "nematic": S_x}, fp)
    else:
        ux = g_vx / safe
        uy = g_vy / safe
        speed = np.sqrt(ux ** 2 + uy ** 2)
        cell_area = (self.edges_x[1] - self.edges_x[0]) * (self.edges_y[1] - self.edges_y[0])
        rho = g_counts / (max(g_counts.sum(), 1) * cell_area)
        temp = mass * g_ke / safe - mass * (ux ** 2 + uy ** 2)
        temp = np.where(g_counts > 1, temp, 0.0)
        S = np.sqrt((g_re / safe) ** 2 + (g_im / safe) ** 2)

        ex, ey = self.edges_x, self.edges_y
        for data, label, suffix in [
            (rho.T, r"$\rho$", "_density"),
            (speed.T, r"$|u|$", "_speed"),
            (temp.T, r"$T$", "_temperature"),
            (S.T, r"$S$", "_nematic"),
        ]:
            fig, ax = plt.subplots(figsize=(8, 4))
            pcm = ax.pcolormesh(ex, ey, data, cmap=pv_cmap, shading="auto", rasterized=True)
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$y$")
            ax.set_aspect("equal")
            ax.tick_params(which="both", direction="in", top=True, right=True)
            cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
            cbar.set_label(label, fontsize=10)
            fig.savefig(f"{prefix}{suffix}.pdf")
            fig.savefig(f"{prefix}{suffix}.png", dpi=400)
            plt.close(fig)

        with open(f"{prefix}_observables.pickle", "wb") as fp:
            pickle.dump({
                "x": 0.5 * (ex[:-1] + ex[1:]),
                "y": 0.5 * (ey[:-1] + ey[1:]),
                "rho": rho, "ux": ux, "uy": uy, "temp": temp, "nematic": S,
            }, fp)


def plot_velocity_histograms(self, prefix=""):
    """2D velocity histogram gathered to rank 0."""
    vel = self.swarm.getField("velocity")
    Vlocal = vel.reshape(self.nlocal, self.dim).copy()
    self.swarm.restoreField("velocity")

    gathered = self.comm.gather(Vlocal, root=0)
    if self.rank != 0:
        return

    V = np.vstack(gathered)

    fig, ax, cax = fig_axes(colorbar=True)
    H, xedges, yedges = np.histogram2d(
        V[:, 0], V[:, 1],
        bins=(self.grid_x, self.grid_y),
    )
    normalisation = H.sum() * (self.delta_x * self.delta_y)
    H = H / normalisation
    pcm = ax.pcolormesh(xedges, yedges, H.T, cmap=pv_cmap, shading="auto", rasterized=True)
    ax.set_xlabel(r"$v_x$")
    ax.set_ylabel(r"$v_y$")
    ax.set_xlim(-self.xlim, self.xlim)
    ax.set_ylim(-self.ylim, self.ylim)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    cbar = fig.colorbar(pcm, cax=cax, format=mticker.ScalarFormatter(useMathText=True))
    cbar.formatter.set_scientific(True)
    cbar.formatter.set_powerlimits((-2, 2))
    cbar.update_ticks()
    cbar.set_label(r"$f(v)$")
    fig.savefig(f"{prefix}_vel.pdf", bbox_inches="tight")
    fig.savefig(f"{prefix}_vel.png", dpi=400, bbox_inches="tight")
    plt.close(fig)
