import matplotlib.pyplot as plt
import numpy as np
import pickle
from dsmc.utils import init_plot, pv_cmap

def plot_history(self, prefix=""):
    fig, ax = plt.subplots(figsize=(5.5, 3.2))  # good for 1-column layout
    # Plot
    time = np.array(self.history["step"])*self.dt
    temp = np.array(self.history["temperature"])
    ax.plot(time,
        temp,
        color="black",
        linewidth=1.5,
    )
    # Labels (no title in SIAM)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$T$")
    # Spines: keep all, but thin and clean (SIAM style often keeps box)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    # Ticks
    ax.tick_params(which="both", direction="in", top=True, right=True)
    # Optional: control limits if needed
    # ax.set_xlim(0, 1)
    # Tight layout
    fig.tight_layout(pad=0.2)
    # Save (high-quality for publication)
    fig.savefig(f"{prefix}_temperature.pdf", bbox_inches="tight")  # PDF preferred
    fig.savefig(f"{prefix}_temperature.png", dpi=400, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.2))  # good for 1-column layout
    # Plot
    var = np.array(self.history["circular_var"])
    ax.plot(time,
        var,
        color="black",
        linewidth=1.5,
    )
    # Labels (no title in SIAM)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\mathrm{Var}(\theta)$")
    # Spines: keep all, but thin and clean (SIAM style often keeps box)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    # Ticks
    ax.tick_params(which="both", direction="in", top=True, right=True)
    # Optional: control limits if needed
    # ax.set_xlim(0, 1)
    # Tight layout
    fig.tight_layout(pad=0.2)
    # Save (high-quality for publication)
    fig.savefig(f"{prefix}_variance.pdf", bbox_inches="tight")  # PDF preferred
    fig.savefig(f"{prefix}_variance.png", dpi=400, bbox_inches="tight")
    plt.close(fig)

def plot_histograms(self, prefix=""):
    step = len(self.history["temperature"])-1
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


    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    # Histogram (normalized!)
    H, xedges, yedges = np.histogram2d(
        V[:, 0], V[:, 1],
        bins=(self.grid_x, self.grid_y),
    )
    normalisation = np.sum(H)*(self.delta_x*self.delta_y)
    H = H/normalisation
    pcm = ax.pcolormesh(
        xedges, yedges, H.T,
        cmap=pv_cmap,
        shading="auto",
        rasterized=True   # nice for vector PDFs
    )
    # Axes
    ax.set_xlabel(r"$v_x$")
    ax.set_ylabel(r"$v_y$")
    ax.set_xlim(-self.xlim, self.xlim)
    ax.set_ylim(-self.ylim, self.ylim)
    # SIAM-style box + ticks
    ax.tick_params(which="both", direction="in", top=True, right=True)
    # Subtle colorbar
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
        with open(f'{prefix}_vel.pickle', 'wb') as fp:
            data = {"hist": H, "xedges": xedges, "yedges": yedges}
            pickle.dump(data, fp)
    #Marginals
    H_x = np.sum(H, axis=1)*self.delta_y
    normalisation = np.sum(H_x)*self.delta_x
    H_x = H_x/normalisation
    # --- Figure (column-width friendly) ---
    fig, ax = plt.subplots(figsize=(5.5, 3.2))  # good for 1-column layout
    # Plot
    ax.plot(
        [x+0.5*self.delta_x for x in xedges[:-1]],
        H_x,
        linestyle="None",
        marker="o",
        markersize=6,
        markerfacecolor="none",
        markeredgecolor="red",
        markeredgewidth=1.0,
    )
    ax.plot(
        xedges,
        Maxwellian[1],
        color="black",
        linewidth=1.5,
    )
    # Labels (no title in SIAM)
    ax.set_xlabel(r"$v_x$")
    ax.set_ylabel(r"$f(v_x)$")
    # Spines: keep all, but thin and clean (SIAM style often keeps box)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    # Ticks
    ax.tick_params(which="both", direction="in", top=True, right=True)
    # Optional: control limits if needed
    # ax.set_xlim(0, 1)
    # Tight layout
    fig.tight_layout(pad=0.2)
    # Save (high-quality for publication)
    fig.savefig(f"{prefix}_vel_x.pdf", bbox_inches="tight")  # PDF preferred
    fig.savefig(f"{prefix}_vel_x.png", dpi=400, bbox_inches="tight")
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    H, thetaedges, omegaedges = np.histogram2d(
        A[:, 0], W[:, 0],
        bins=(self.grid_angular, self.grid_omega),
    )
    normalisation = np.sum(H)*(self.delta_angular*self.delta_omega)
    H = H/normalisation
    pcm = ax.pcolormesh(
        thetaedges, omegaedges, H.T,
        cmap=pv_cmap,
        shading="auto",
        rasterized=True   # nice for vector PDFs
    )
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
        with open(f'{prefix}_angular.pickle', 'wb') as fp:
            data = {"hist_theta": A, "hist_omega": W, "theta_edges": thetaedges, "omegaedges": omegaedges}
            pickle.dump(data, fp)
    #Marginals
    H_theta = np.sum(H, axis=1)*self.delta_omega
    normalisation = np.sum(H_theta)*self.delta_omega
    H_theta = H_theta/normalisation
    # --- Figure (column-width friendly) ---
    fig, ax = plt.subplots(figsize=(5.5, 3.2))  # good for 1-column layout
    # Plot
    ax.plot(
        [theta+0.5*self.delta_angular for theta in thetaedges[:-1]],
        H_theta,
        linestyle="None",
        marker="o",
        markersize=6,
        markerfacecolor="none",
        markeredgecolor="red",
        markeredgewidth=1.0,
    )
    # Labels (no title in SIAM)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$f(\theta)$")
    m = np.mean(H_theta)
    ymin = min(H_theta.min(), m - 0.1)
    ymax = max(H_theta.max(), m + 0.1)
    margin = 0.02 * (ymax - ymin)
    ax.set_ylim(ymin - margin, ymax + margin)
    # Spines: keep all, but thin and clean (SIAM style often keeps box)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    # Ticks
    ax.tick_params(which="both", direction="in", top=True, right=True)
    # Optional: control limits if needed
    # ax.set_xlim(0, 1)
    # Tight layout
    fig.tight_layout(pad=0.2)
    # Save (high-quality for publication)
    fig.savefig(f"{prefix}_theta.pdf", bbox_inches="tight")  # PDF preferred
    fig.savefig(f"{prefix}_theta.png", dpi=400, bbox_inches="tight")
    plt.close(fig)
