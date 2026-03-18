import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

def init_plot():
    # --- SIAM style ---
    mpl.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",        # Computer Modern (LaTeX-like)
        "font.size": 10,                 # SIAM papers are compact
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


def paraview_cool_to_warm_extended():
    # Sampled from ParaView's "Cool to Warm (Extended)" preset
    pts = [
        (0.000000, (0.000000, 0.000000, 0.349020)),
        (0.031250, (0.039216, 0.062745, 0.380392)),
        (0.062500, (0.062745, 0.117647, 0.411765)),
        (0.093750, (0.090196, 0.184314, 0.450980)),
        (0.125000, (0.125490, 0.262745, 0.501961)),
        (0.156250, (0.160784, 0.337255, 0.541176)),
        (0.187500, (0.200000, 0.396078, 0.568627)),
        (0.218750, (0.239216, 0.454902, 0.600000)),
        (0.250000, (0.286275, 0.521569, 0.650980)),
        (0.281250, (0.337255, 0.592157, 0.701961)),
        (0.312500, (0.388235, 0.654902, 0.749020)),
        (0.343750, (0.466667, 0.737255, 0.819608)),
        (0.375000, (0.572549, 0.819608, 0.878431)),
        (0.406250, (0.654902, 0.866667, 0.909804)),
        (0.437500, (0.752941, 0.917647, 0.941176)),
        (0.468750, (0.823529, 0.956863, 0.968627)),
        (0.500000, (0.941176, 0.984314, 0.988235)),
        (0.500000, (0.988235, 0.960784, 0.901961)),
        (0.520000, (0.988235, 0.945098, 0.850980)),
        (0.540000, (0.980392, 0.898039, 0.784314)),
        (0.562500, (0.968627, 0.835294, 0.698039)),
        (0.593750, (0.949020, 0.733333, 0.588235)),
        (0.625000, (0.929412, 0.650980, 0.509804)),
        (0.656250, (0.909804, 0.564706, 0.435294)),
        (0.687500, (0.878431, 0.458824, 0.352941)),
        (0.718750, (0.839216, 0.388235, 0.286275)),
        (0.750000, (0.760784, 0.294118, 0.211765)),
        (0.781250, (0.701961, 0.211765, 0.168627)),
        (0.812500, (0.650980, 0.156863, 0.129412)),
        (0.843750, (0.600000, 0.094118, 0.094118)),
        (0.875000, (0.549020, 0.066667, 0.098039)),
        (0.906250, (0.501961, 0.050980, 0.125490)),
        (0.937500, (0.450980, 0.054902, 0.172549)),
        (0.968750, (0.400000, 0.054902, 0.192157)),
        (1.000000, (0.349020, 0.070588, 0.211765)),
    ]
    return LinearSegmentedColormap.from_list("pv_cool_to_warm_extended", pts, N=256)

pv_cmap = paraview_cool_to_warm_extended()

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

    gathered = self.comm.gather(Vlocal, root=0)
    if self.rank != 0:
        return

    V = np.vstack(gathered)

    angle = self.swarm.getField("orientation")
    Alocal = angle.reshape(self.nlocal, 1).copy()
    self.swarm.restoreField("orientation")

    gathered = self.comm.gather(Alocal, root=0)
    if self.rank != 0:
        return

    A = np.vstack(gathered)

    omega = self.swarm.getField("angular_velocity")
    Wlocal = omega.reshape(self.nlocal, 1).copy()
    self.swarm.restoreField("angular_velocity")

    gathered = self.comm.gather(Wlocal, root=0)
    if self.rank != 0:
        return

    W = np.vstack(gathered)


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
    np.save(f"{prefix}_vel.npy", V)
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
    np.save(f"{prefix}_theta.npy", A)
    np.save(f"{prefix}_omega.npy", W)
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
