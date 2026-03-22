import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pickle
from mpi4py import MPI


def init_plot():
    mpl.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
    })


def plot_observables(self, prefix=""):
    """
    Plot spatial profiles of density, mean velocity, and temperature.
    Uses allreduce so must be called on all ranks.
    """
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

    # Per-cell local sums (all ranks)
    local_counts = np.histogram(xpos, bins=edges)[0].astype(float)
    local_vel_sum = np.histogram(xpos, bins=edges, weights=V[:, 0])[0]
    local_ke_sum = np.histogram(xpos, bins=edges,
                                weights=V[:, 0] ** 2 + V[:, 1] ** 2)[0]

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
    """
    2-D velocity histogram gathered to rank 0.
    Must be called on all ranks.
    """
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
    pcm = ax.pcolormesh(xedges, yedges, H.T, shading="auto", rasterized=True)
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


def plot_history(self, prefix=""):
    """Plot temperature and energy history. Call from rank 0 only."""
    time = np.array(self.history["step"]) * self.dt

    for data, ylabel, fname_suffix in [
        (self.history["temperature"], r"$T$", "_temperature"),
        (self.history["energy"], r"$E$", "_energy"),
    ]:
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        ax.plot(time, np.array(data), color="black")
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(ylabel)
        ax.tick_params(which="both", direction="in", top=True, right=True)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
        fig.tight_layout(pad=0.2)
        fig.savefig(f"{prefix}{fname_suffix}.pdf", bbox_inches="tight")
        fig.savefig(f"{prefix}{fname_suffix}.png", dpi=400, bbox_inches="tight")
        plt.close(fig)
