#!/usr/bin/env python3
import petsc4py
import numpy as np
import sys
import os
#passing system argument to PETSc
petsc4py.init(sys.argv)
from petsc4py import PETSc
from mpi4py import MPI
import gc
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
from graphics import pv_cmap

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

def Print(*args, **kwargs):
    """Helper to print only from rank 0."""
    if PETSc.COMM_WORLD.Get_rank() == 0:
        print(*args, **kwargs)

def _build_cell_lists(cells):

    if cells.size == 0:
        return {}

    order = np.argsort(cells)
    cells_sorted = cells[order]

    # starts of each new cell block
    starts = np.flatnonzero(
        np.r_[True, cells_sorted[1:] != cells_sorted[:-1]]
    )
    ends = np.r_[starts[1:], len(cells_sorted)]

    cell_lists = {}
    for a, b in zip(starts, ends):
        c = int(cells_sorted[a])
        cell_lists[c] = order[a:b]

    return cell_lists

class CFMDSMC:
    """
    DSMC for space-inhomogeneous CFM kinetic equation.

    Particles carry only velocity v in R^2.
    Collisions are Bird/Nanbu-style random pair collisions with
    isotropic post-collisional direction.

    For Maxwell molecules, the collision kernel does not depend on |g|,
    so pair selection is uniform.
    """

    def __init__(
        self,
        nlocal: int,
        nu: float = 1.0,
        dt: float = 1e-2,
        info: dict = {},
        extra_collision: int = 1,
        grazing_collision: bool = False,
        collision_type: str = "nanbu",
        seed: int = 1234,
        bins: int = 31,
        test: str = "sod",
        comm: MPI.Comm = MPI.COMM_WORLD,
    ): 
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

        self.dim = 2
        self.nlocal = nlocal
        self.N = self.nlocal * self.size
        self.nu = nu 
        self.dt = dt
        if 1/nu > dt:
            raise RuntimeError("You have too large of a time-step for the collisional frequency you specified")
        self.info = info
        self.bins = bins
        self.delta_bins = 1/(bins+1)
        self.test = test
        self.extra_collision = extra_collision
        self.grazing_collision = grazing_collision
        self.collision_type = collision_type


        self.xlim = 10.0
        self.ylim = 10.0

        self.rng = np.random.default_rng(seed + self.rank)

        # History of simulation
        self.history = {
            "step": [],
            "temperature": [],
            "energy": [],
            "momentum_1": [],
            "momentum_2": [],
            "ang_momentum": [],
        }

        self.output_path = f'output_cfm_{self.collision_type}' 
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        self.dm = self._create_mesh()
        self.mesh_dim = self.dm.getDimension()

        self.swarm = self._create_swarm()
        self._initialize_particles()

    def _create_mesh(self):
        nx = self.bins
        self.edges_x = np.linspace(0.0, 2*np.pi, nx + 1)
        dm = PETSc.DMDA().create([nx+1, 2], dof=1, stencil_width=1, comm=self.comm)
        dm.setUp()
        dm.setUniformCoordinates(0.0, 2*np.pi, 0.0, 1.0)
        return dm

    def _create_swarm(self):
        swarm = PETSc.DMSwarm().create(comm=self.comm)
        # For this use-case we only need DMSwarm as a generic particle container.
        # In most PETSc builds the BASIC type is appropriate here.
        swarm.setDimension(self.mesh_dim)
        swarm.setType(PETSc.DMSwarm.Type.PIC)
        swarm.setCellDM(self.dm)

        swarm.initializeFieldRegister()
        swarm.registerField("orientation", 1, dtype=PETSc.RealType)
        swarm.registerField("velocity", self.dim, dtype=PETSc.RealType)
        swarm.registerField("angular_velocity", 1, dtype=PETSc.RealType)
        swarm.registerField("weight", 1, dtype=PETSc.RealType)
        swarm.finalizeFieldRegister()

        # buffer > 0 is handy if later you want insertion/removal
        swarm.setLocalSizes(self.nlocal, max(16, self.nlocal // 10))
        return swarm

    def _initialize_particles(self):
        """
        Initialize with a Maxwellian N(0, T/m I).
        """

        angle = self.swarm.getField("orientation")
        vel = self.swarm.getField("velocity")
        angular_vel = self.swarm.getField("angular_velocity")
        wgt = self.swarm.getField("weight")
        angle[:] = self.rng.uniform(size=(self.nlocal, 1), low=0, high=2*np.pi)
        vel[:] = self.rng.uniform(size=(self.nlocal, 2), low=-2, high=2)
        angular_vel[:] = self.rng.uniform(size=(self.nlocal, 1), low=-1, high=1)
        wgt[:] = 1.0
        self.swarm.restoreField("orientation")
        self.swarm.restoreField("velocity")
        self.swarm.restoreField("angular_velocity")
        self.swarm.restoreField("weight")

        #Change value for graphs
        self.xlim = 8.0
        self.ylim = 8.0
        self.angular_min = 0.0
        self.angular_max = 2*np.pi
        self.omega_min = -8.0
        self.omega_max = 8.0
        if self.angular_min > self.angular_max:
            raise RunTimeError("[!] Larger angular min than angular max.")
        if self.omega_min > self.omega_max:
            raise RunTimeError("[!] Larger omega min than omega max.")
    def _construct_grid(self):

        grid_x = np.linspace(-self.xlim, self.xlim, self.bins + 1)
        grid_y = np.linspace(-self.ylim, self.ylim, self.bins + 1)
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.delta_x = (2*self.xlim)/(self.bins+1)
        self.delta_y = (2*self.ylim)/(self.bins+1)
         
        grid_angular = np.linspace(self.angular_min, self.angular_max, self.bins + 1)
        grid_omega = np.linspace(self.omega_min, self.omega_max, self.bins + 1)
        self.grid_angular = grid_angular
        self.grid_omega = grid_omega
        self.delta_angular = (self.angular_max-self.angular_min)/(self.bins+1)
        self.delta_omega = (self.omega_max-self.omega_min)/(self.bins+1)


    def diagnostics(self, step=0):
        angle = self.swarm.getField("orientation")
        vel = self.swarm.getField("velocity")
        omega = self.swarm.getField("angular_velocity")

        if sum(np.where(angle >= 2*np.pi,1,0))>0:
            raise RuntimeError("[!] Not sticking to the manifold!")
        if sum(np.where(angle <= 0, 1,0))>0:
            raise RuntimeError("[!] Not sticking to the manifold!")

        local_n = self.nlocal
        global_n = self.comm.allreduce(local_n, op=MPI.SUM)

        local_mom = vel.sum(axis=0)
        global_mom = self.comm.allreduce(local_mom, op=MPI.SUM)

        local_ang_mom = omega.sum(axis=0) 
        global_ang_mom = self.comm.allreduce(local_ang_mom, op=MPI.SUM)

        local_energy = 0.5 * self.info["mass"] * np.sum(vel*vel) + 0.5*self.info["inertia"]*np.sum(omega*omega)
        global_energy = self.comm.allreduce(local_energy, op=MPI.SUM)

        mean_u = global_mom / global_n
        mean_eta = global_ang_mom/global_n
        temp = (2.0 / (self.dim+1)) * global_energy / global_n 
        
        self.history["step"].append(step)
        self.history["temperature"].append(temp)
        self.history["momentum_1"].append(np.linalg.norm(mean_u[0]))
        self.history["momentum_2"].append(np.linalg.norm(mean_u[1]))
        self.history["ang_momentum"].append(np.linalg.norm(mean_eta))

        self.swarm.restoreField("orientation")
        self.swarm.restoreField("velocity")
        self.swarm.restoreField("angular_velocity")
        return {
            "N": global_n,
            "mean_u": mean_u,
            "temperature": temp,
        }

    def _sample_angle(self, m: int):
        Theta = self.rng.uniform(size=(m, 1)) * 2 * np.pi
        return Theta

    def _get_particle_cells(self):
        celldm = self.swarm.getCellDMActive()
        cellid_name = celldm.getCellID()

        arr = self.swarm.getField(cellid_name)
        try:
            out = np.asarray(arr).copy()
            return out.reshape(-1).astype(np.int32)
        finally:
             self.swarm.restoreField(cellid_name)

    def nanbu_collision_step(self):
        vel = self.swarm.getField("velocity").reshape(self.nlocal, self.dim)
        theta = self.swarm.getField("orientation").reshape(self.nlocal)
        omega = self.swarm.getField("angular_velocity").reshape(self.nlocal)

        # number of collision pairs
        Mcol = int(0.5 * self.nu * self.nlocal * self.dt)
        Mcol = Mcol if Mcol % 2 == 0 else Mcol - 1
        if 2 * Mcol > self.nlocal:
            Mcol = Mcol - 2
        if Mcol <= 0:
            self.swarm.restoreField("velocity")
            self.swarm.restoreField("orientation")
            self.swarm.restoreField("angular_velocity")
            return

        # random permutation and pairing
        r = self.rng.permutation(self.nlocal)
        i = r[:Mcol]
        j = r[Mcol:2 * Mcol]

        vi = vel[i].copy()
        vj = vel[j].copy()

        thetai = theta[i].copy()
        thetaj = theta[j].copy()

        omegai = omega[i].copy()
        omegaj = omega[j].copy()

        # rod directions
        nui = np.column_stack((np.cos(thetai), np.sin(thetai)))
        nuj = np.column_stack((np.cos(thetaj), np.sin(thetaj)))

        # random impact angle
        psi = 2.0 * np.pi * self.rng.random(Mcol)
        n = np.column_stack((np.cos(psi), np.sin(psi)))
        n_cutoff = n.copy()

        # cutoff for near-parallel rods
        dtheta = np.abs(thetai - thetaj)
        cutoff = self.info.get("cutoff", 0.1)
        idx = (dtheta > cutoff) & (dtheta < 2.0 * np.pi - cutoff)
        idxf = idx.astype(float)
        not_idxf = 1.0 - idxf

        # sampled contact arms
        L = self.info["length"]
        ell = L * self.rng.random(Mcol)

        ri = ell[:, None] * nui
        rj = L * nuj   # this matches your MATLAB code literally

        # relative contact velocity
        ri_perp = np.column_stack((ri[:, 1], -ri[:, 0]))
        rj_perp = np.column_stack((rj[:, 1], -rj[:, 0]))

        V = vi - vj + omegai[:, None] * ri_perp - omegaj[:, None] * rj_perp

        # impulse
        m = self.info["mass"]
        I = self.info["inertia"]

        ci = ri[:, 0] * n[:, 1] - ri[:, 1] * n[:, 0]
        cj = rj[:, 0] * n[:, 1] - rj[:, 1] * n[:, 0]

        denom = 2.0 / m + (ci**2 + cj**2) / I
        J = -np.sum(V * n, axis=1) / denom

        # restitution coefficients
        ev = self.info.get("ev", 1.0)
        eom = self.info.get("om", 1.0)

        # fallback spherical-like collision for nearly parallel rods
        vn_cut = np.sum((vi - vj) * n_cutoff, axis=1)

        vi_prime = (
            vi
            + ((1.0 + ev) * J / m)[:, None] * n * idxf[:, None]
            - 0.5 * (1.0 + ev) * vn_cut[:, None] * n_cutoff * not_idxf[:, None]
        )

        vj_prime = (
            vj
            - ((1.0 + ev) * J / m)[:, None] * n * idxf[:, None]
            + 0.5 * (1.0 + ev) * vn_cut[:, None] * n_cutoff * not_idxf[:, None]
        )

        omegai_prime = omegai - (1.0 + eom) * J * ci / I * idxf
        omegaj_prime = omegaj + (1.0 + eom) * J * cj / I * idxf

        # write back
        vel[i] = vi_prime
        vel[j] = vj_prime
        omega[i] = omegai_prime
        omega[j] = omegaj_prime

        self.swarm.restoreField("velocity")
        self.swarm.restoreField("orientation")
        self.swarm.restoreField("angular_velocity")
    """
    def _reflect_1d(self, x, v, xmin, xmax):
        L = xmax - xmin
        I_left = np.where(x < xmin)
        I_right = np.where(x > xmax)
        x[I_left] = -np.abs(x[I_left])*np.sign(v[I_left])
        x[I_right] = L - np.abs(x[I_right]-xmax)*np.sign(v[I_right])
        v[I_left] = -v[I_left]
        v[I_right] = -v[I_right]
    """
    def _stick_to_manifold(self, angle):
        return np.mod(angle, 2*np.pi)

    def transport_step(self, dt):
        angle = self.swarm.getField("orientation")
        omega = self.swarm.getField("angular_velocity")

        angle[:, 0] += omega[:, 0] * dt
        angle[:,0] = self._stick_to_manifold(angle[:, 0])

        self.swarm.restoreField("orientation")
        self.swarm.restoreField("angular_velocity")
    
    def maxwellian(self, step):
        vx, vy, omega = np.meshgrid(self.grid_x,self.grid_y, self.grid_omega)
        I = self.info["inertia"]
        m = self.info["mass"]
        temp = self.history["temperature"][step]
        momentum_1 = self.history["momentum_1"][step] 
        momentum_2 = self.history["momentum_2"][step] 
        ang_momentum = self.history["ang_momentum"][step]
        normalisation = m*np.sqrt(I/((2*np.pi*temp)**3))
        Maxwellian = normalisation*np.exp(-(0.5*m/temp)*(vx-momentum_1)**2)
        Maxwellian = Maxwellian*np.exp(-(0.5*m/temp)*(vy-momentum_2)**2)
        Maxwellian = Maxwellian*np.exp(-(0.5*I/temp)*(omega-ang_momentum)**2)
        Maxwellian_x = np.sum(Maxwellian, axis=(1,2))*self.delta_y*self.delta_omega
        #Maxwellian_x = 1/np.sqrt(2*np.pi*temp)*np.exp(-(0.5/temp)*self.grid_x**2)
        Maxwellian_y = np.sum(Maxwellian, axis=(0,2))*self.delta_x*self.delta_omega
        Maxwellian_omega = np.sum(Maxwellian, axis=(0,1))*self.delta_x*self.delta_y
        #import pdb; pdb.set_trace()
        return Maxwellian, Maxwellian_x, Maxwellian_y, Maxwellian_omega

    def plot_histograms(self, prefix=""):
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
        H_x, xedges = np.histogram(V[:, 0], bins=self.grid_x)
        normalisation = np.sum(H_x)*self.delta_x
        H_x = H_x/normalisation
        print(np.sum(H_x)*self.delta_x)
        #import pdb; pdb.set_trace()
        # --- Figure (column-width friendly) ---
        fig, ax = plt.subplots(figsize=(5.5, 3.2))  # good for 1-column layout
        # Plot
        ax.plot(
            xedges[:-1],
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
        H, xedges, yedges = np.histogram2d(
            A[:, 0], W[:, 0],
            bins=(self.grid_angular, self.grid_omega),
        )
        normalisation = np.sum(H)*(self.delta_angular*self.delta_omega)
        H = H/normalisation
        pcm = ax.pcolormesh(
            xedges, yedges, H.T,
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

    def run(self, nsteps: int, monitor_every: int = 10):
        if self.rank == 0:
            d = self.diagnostics()
            print(
                f"[step 0] N={d['N']}, t = 0.0 "
                f"T={d['temperature']:.6e} "
                f"|u|={np.linalg.norm(d['mean_u']):.6e} "
            )
        self._construct_grid()
        if self.rank == 0:
            self.plot_histograms(prefix=f"{self.output_path}/dsmc_0")
        for step in range(1, nsteps + 1):
            self.transport_step(dt=0.5*self.dt)
            for coll_index in range(self.extra_collision):
                if self.collision_type == "nanbu":
                    self.nanbu_collision_step()
                else:
                    raise ValueError(f"Unknown collision type: {self.collision_type}")
            self.transport_step(dt=0.5*self.dt)
            d = self.diagnostics(step=step)
            if self.rank == 0:
                print(
                    f"[step {step}] N={d['N']}, t ={step*self.dt:.6f} "
                    f"T={d['temperature']:.6e} "
                    f"|u|={np.linalg.norm(d['mean_u']):.6e} "
                )
                if step % monitor_every == 0 or step == nsteps:
                    self.plot_histograms(prefix=f"{self.output_path}/dsmc_{step}")
            gc.collect()


def main():
    Opt = PETSc.Options()
    Print("Running homogeneous Maxwell DSMC with options:")

    nlocal = Opt.getReal("nlocal", 20000)
    nlocal = int(nlocal)
    bins = Opt.getInt("bins", 31)
    dt = Opt.getReal("dt", 1e-2)
    nu = Opt.getReal("nu", 1/dt)
    nsteps = Opt.getInt("nsteps", 200)
    seed = Opt.getInt("seed", 1234)
    grazing_collision = Opt.getBool("grazing_collision", False)
    collision_type = Opt.getString("collision_type", "nanbu")
    extra_collision = Opt.getInt("extra_collision", 0)+1
    monitor_every = Opt.getInt("monitor_every", 10)

    Print(f"  nlocal={nlocal}")
    Print(f"  nu={nu}")
    Print(f"  dt={dt}")
    Print(f"  collision ratio is {nu*dt}")
    Print(f"  bins={bins}")
    Print(f"  nsteps={nsteps}")
    Print(f"  seed={seed}")
    Print(f"  monitor_every={monitor_every}")
    Print(f"  extra_collision={extra_collision}")
    Print(f"  collision_type={collision_type}")
    Print(f"  grazing_collision={grazing_collision}")

    Print("--------------------------------------------------------------------")
    
    #TODO: Fix with correct relation between length mass and inertia
    info = {"inertia": 1.0,
            "mass": 1.0,
            "length": 1.0,
            "ev": 1.0,       # translational restitution
            "om": 1.0,       # rotational restitution
            "cutoff": 0.1,   # angular cutoff
           }

    sim = CFMDSMC(
        nlocal=nlocal,
        nu=nu,
        dt=dt,
        bins = bins,
        info=info,
        extra_collision=extra_collision,
        grazing_collision=grazing_collision,
        collision_type=collision_type,
        seed=seed,
        comm=MPI.COMM_WORLD,
    )
    sim.run(nsteps=nsteps, monitor_every=monitor_every)
    Print("Simulation complete.")

if __name__ == "__main__":
    main()
