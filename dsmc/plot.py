import matplotlib.pyplot as plt
import numpy as np
import pickle
from dsmc.utils import init_plot, pv_cmap


def plot_history(self, prefix=""):
    time = np.array(self.history["step"]) * self.dt

    to_plot = []
    if "temperature" in self.history:
        to_plot.append((self.history["temperature"], r"$T$", "_temperature"))
    if "energy" in self.history:
        to_plot.append((self.history["energy"], r"$E$", "_energy"))
    if "circular_var" in self.history:
        to_plot.append((self.history["circular_var"], r"$\mathrm{Var}(\theta)$", "_variance"))

    for data, ylabel, fname_suffix in to_plot:
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        ax.plot(time, np.array(data), color="black", linewidth=1.5)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(ylabel)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
        ax.tick_params(which="both", direction="in", top=True, right=True)
        fig.tight_layout(pad=0.2)
        fig.savefig(f"{prefix}{fname_suffix}.pdf", bbox_inches="tight")
        fig.savefig(f"{prefix}{fname_suffix}.png", dpi=400, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# CFM-specific
# ---------------------------------------------------------------------------

def plot_histograms(self, prefix=""):
    step = len(self.history["temperature"]) - 1
    if self.vlasov_force:
        Maxwellian = self.maxwellian(step)
    else:
        Maxwellian = self.maxwellian(0)

    vel = self.swarm.getField("velocity")
    Vlocal = vel.reshape(self.nlocal, self.dim).copy()
    self.swarm.restoreField("velocity")

    angle = self.swarm.getField("orientation")
    Alocal = angle.reshape(self.nlocal, 1).copy()
    self.swarm.restoreField("orientation")

    omega = self.swarm.getField("angular_velocity")
    Wlocal = omega.reshape(self.nlocal, 1).copy()
    self.swarm.restoreField("angular_velocity")

    V_gathered = self.comm.gather(Vlocal, root=0)
    A_gathered = self.comm.gather(Alocal, root=0)
    W_gathered = self.comm.gather(Wlocal, root=0)

    if self.rank != 0:
        return

    V = np.vstack(V_gathered)
    A = np.vstack(A_gathered)
    W = np.vstack(W_gathered)

    # --- 2D velocity histogram ---
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    H, xedges, yedges = np.histogram2d(
        V[:, 0], V[:, 1],
        bins=(self.grid_x, self.grid_y),
    )
    normalisation = np.sum(H) * (self.delta_x * self.delta_y)
    H = H / normalisation
    pcm = ax.pcolormesh(xedges, yedges, H.T, cmap=pv_cmap, shading="auto", rasterized=True)
    ax.set_xlabel(r"$v_x$")
    ax.set_ylabel(r"$v_y$")
    ax.set_xlim(-self.xlim, self.xlim)
    ax.set_ylim(-self.ylim, self.ylim)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(r"$f(v)$", fontsize=10)
    fig.tight_layout(pad=0.2)
    fig.savefig(f"{prefix}_vel.pdf", bbox_inches="tight")
    fig.savefig(f"{prefix}_vel.png", dpi=400, bbox_inches="tight")
    plt.close(fig)
    if self.dump == "particles":
        np.save(f"{prefix}_vel.npy", V)
    elif self.dump == "hist":
        with open(f"{prefix}_vel.pickle", "wb") as fp:
            pickle.dump({"hist": H, "xedges": xedges, "yedges": yedges}, fp)

    # --- vx marginal ---
    H_x = np.sum(H, axis=1) * self.delta_y
    H_x = H_x / (np.sum(H_x) * self.delta_x)
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(
        [x + 0.5 * self.delta_x for x in xedges[:-1]], H_x,
        linestyle="None", marker="o", markersize=6,
        markerfacecolor="none", markeredgecolor="red", markeredgewidth=1.0,
        label="DSMC",
    )
    ax.plot(xedges, Maxwellian[1], color="black", linewidth=1.5, label="Maxwellian")
    ax.set_xlabel(r"$v_x$")
    ax.set_ylabel(r"$f(v_x)$")
    ax.legend()
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    fig.tight_layout(pad=0.2)
    fig.savefig(f"{prefix}_vel_x.pdf", bbox_inches="tight")
    fig.savefig(f"{prefix}_vel_x.png", dpi=400, bbox_inches="tight")
    plt.close(fig)

    # --- vy marginal ---
    H_y = np.sum(H, axis=0) * self.delta_x
    H_y = H_y / (np.sum(H_y) * self.delta_y)
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(
        [y + 0.5 * self.delta_y for y in yedges[:-1]], H_y,
        linestyle="None", marker="o", markersize=6,
        markerfacecolor="none", markeredgecolor="red", markeredgewidth=1.0,
        label="DSMC",
    )
    ax.plot(yedges, Maxwellian[2], color="black", linewidth=1.5, label="Maxwellian")
    ax.set_xlabel(r"$v_y$")
    ax.set_ylabel(r"$f(v_y)$")
    ax.legend()
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    fig.tight_layout(pad=0.2)
    fig.savefig(f"{prefix}_vel_y.pdf", bbox_inches="tight")
    fig.savefig(f"{prefix}_vel_y.png", dpi=400, bbox_inches="tight")
    plt.close(fig)

    # --- 2D angular histogram ---
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    H, thetaedges, omegaedges = np.histogram2d(
        A[:, 0], W[:, 0],
        bins=(self.grid_angular, self.grid_omega),
    )
    normalisation = np.sum(H) * (self.delta_angular * self.delta_omega)
    H = H / normalisation
    pcm = ax.pcolormesh(thetaedges, omegaedges, H.T, cmap=pv_cmap, shading="auto", rasterized=True)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\omega$")
    ax.set_xlim(self.angular_min, self.angular_max)
    ax.set_ylim(self.omega_min, self.omega_max)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(r"$f(\theta,\omega)$", fontsize=10)
    fig.tight_layout(pad=0.2)
    fig.savefig(f"{prefix}_angular.pdf", bbox_inches="tight")
    fig.savefig(f"{prefix}_angular.png", dpi=400, bbox_inches="tight")
    plt.close(fig)
    if self.dump == "particles":
        np.save(f"{prefix}_theta.npy", A)
        np.save(f"{prefix}_omega.npy", W)
    elif self.dump == "hist":
        with open(f"{prefix}_angular.pickle", "wb") as fp:
            pickle.dump({"hist_theta": A, "hist_omega": W, "theta_edges": thetaedges, "omegaedges": omegaedges}, fp)

    # --- theta marginal ---
    H_theta = np.sum(H, axis=1) * self.delta_omega
    H_theta = H_theta / (np.sum(H_theta) * self.delta_angular)
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(
        [theta + 0.5 * self.delta_angular for theta in thetaedges[:-1]], H_theta,
        linestyle="None", marker="o", markersize=6,
        markerfacecolor="none", markeredgecolor="red", markeredgewidth=1.0,
        label="DSMC",
    )
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$f(\theta)$")
    ax.legend()
    m = np.mean(H_theta)
    ymin = min(H_theta.min(), m - 0.1)
    ymax = max(H_theta.max(), m + 0.1)
    margin = 0.02 * (ymax - ymin)
    ax.set_ylim(ymin - margin, ymax + margin)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    fig.tight_layout(pad=0.2)
    fig.savefig(f"{prefix}_theta.pdf", bbox_inches="tight")
    fig.savefig(f"{prefix}_theta.png", dpi=400, bbox_inches="tight")
    plt.close(fig)

    # --- omega marginal ---
    H_omega = np.sum(H, axis=0) * self.delta_angular
    H_omega = H_omega / (np.sum(H_omega) * self.delta_omega)
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
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
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    fig.tight_layout(pad=0.2)
    fig.savefig(f"{prefix}_omega.pdf", bbox_inches="tight")
    fig.savefig(f"{prefix}_omega.png", dpi=400, bbox_inches="tight")
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
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        ax.plot(x_centers, data, color="black")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(ylabel)
        ax.tick_params(which="both", direction="in", top=True, right=True)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
        fig.tight_layout(pad=0.2)
        fig.savefig(fname, bbox_inches="tight")
        fig.savefig(fname.replace(".pdf", ".png"), dpi=400, bbox_inches="tight")
        plt.close(fig)

    with open(f"{prefix}_observables.pickle", "wb") as fp:
        pickle.dump({"x": x_centers, "rho": rho_x, "vel_x": vel_x, "temp": temp_x}, fp)


def plot_velocity_histograms(self, prefix=""):
    """2D velocity histogram gathered to rank 0."""
    vel = self.swarm.getField("velocity")
    Vlocal = vel.reshape(self.nlocal, self.dim).copy()
    self.swarm.restoreField("velocity")

    gathered = self.comm.gather(Vlocal, root=0)
    if self.rank != 0:
        return

    V = np.vstack(gathered)

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
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
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(r"$f(v)$", fontsize=10)
    fig.tight_layout(pad=0.2)
    fig.savefig(f"{prefix}_vel.pdf", bbox_inches="tight")
    fig.savefig(f"{prefix}_vel.png", dpi=400, bbox_inches="tight")
    plt.close(fig)
