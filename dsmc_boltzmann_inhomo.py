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
from numba import jit


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

class MaxwellDSMC:
    """
    DSMC for space-inhomogeneous Maxwell molecules.

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
        temperature: float = 1.0,
        mass: float = 1.0,
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
        if test == "sod":
            self.effective_dim = 1
        else:
            self.effective_dim = self.dim
        self.nlocal = nlocal
        self.N = self.nlocal * self.size
        self.nu = nu              # collision frequency
        self.dt = dt
        self.temperature = temperature
        self.mass = mass
        self.bins = bins
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
        }

        self.output_path = f'output_boltzmann_{self.collision_type}' 
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        self.dm = self._create_mesh()
        self.mesh_dim = self.dm.getDimension()

        self.swarm = self._create_swarm()
        self._initialize_particles()

    def _create_mesh(self):
        if self.test == "sod":
            Lx = 1.0
            self.info = {"Lx": Lx}
            nx = self.bins
            self.edges_x = np.linspace(0.0, Lx, nx + 1)
            dim = 2
            dm = PETSc.DMDA().create([nx+1, 2], dof=1, stencil_width=1, comm=self.comm)
            dm.setUp()
            dm.setUniformCoordinates(0.0, Lx, 0.0, 1.0)
        return dm

    def _create_swarm(self):
        swarm = PETSc.DMSwarm().create(comm=self.comm)
        # For this use-case we only need DMSwarm as a generic particle container.
        # In most PETSc builds the BASIC type is appropriate here.
        swarm.setDimension(self.mesh_dim)
        swarm.setType(PETSc.DMSwarm.Type.PIC)
        swarm.setCellDM(self.dm)

        swarm.initializeFieldRegister()
        swarm.registerField("velocity", self.dim, dtype=PETSc.RealType)
        swarm.registerField("weight", 1, dtype=PETSc.RealType)
        swarm.finalizeFieldRegister()

        # buffer > 0 is handy if later you want insertion/removal
        swarm.setLocalSizes(self.nlocal, max(16, self.nlocal // 10))
        return swarm

    def _initialize_particles(self):
        """
        Initialize with a Maxwellian N(0, T/m I).
        """
        sigma = np.sqrt(0.5*self.temperature / self.mass)
        if self.test == "sod":
            self.info["norm_rho"] = 0.5*(1+0.125)
            X = np.zeros((self.nlocal, self.mesh_dim))
            n_left = int(0.89*self.nlocal)
            n_right = self.nlocal - n_left
            X[:n_left, 0] = self.rng.uniform(0.0, 0.5*self.info["Lx"], size=n_left)
            X[n_left:, 0] = self.rng.uniform(0.5, 1*self.info["Lx"], size=n_right)
            X[:,0] = np.sort(X[:,0]) 
            X[:,1] = 0.5
            self.swarm.setPointCoordinates(X)
            self.swarm.migrate(remove_sent_points=False)

            celldm = self.swarm.getCellDMActive()
            cellid_name = celldm.getCellID()
            cellid = self.swarm.getField(cellid_name)
            vel = self.swarm.getField("velocity")
            wgt = self.swarm.getField("weight")
            counts = np.bincount(cellid[:,0])
            # isotropic Gaussian
            tmp_index = 0
            for i in range(self.bins):
                count = counts[i]
                gaussian = self.rng.normal(size=(count, self.dim))
                v_prime = np.sum(gaussian, axis=0)/count
                #Smoothing
                v_exact = 0; e_exact = 1
                v_prime = np.sum(gaussian, axis=0)/count
                e_prime = np.sum(gaussian**2, axis=0)/count
                tau = np.sqrt((e_prime-v_prime**2)/(e_exact-v_exact**2))
                lam = v_prime - tau* v_exact
                gaussian = (gaussian - lam)/tau
                #Smoothing twice for better results
                v_prime = np.sum(gaussian, axis=0)/count
                e_prime = np.sum(gaussian**2, axis=0)/count
                tau = np.sqrt((e_prime-v_prime**2)/(e_exact-v_exact**2))
                lam = v_prime - tau* v_exact
                gaussian = (gaussian - lam)/tau
                if count == 0:
                    continue
                if i > self.bins//2:
                    vel[tmp_index:tmp_index+count, :] = gaussian * np.sqrt(0.8) * sigma
                else:
                    vel[tmp_index:tmp_index+count, :] = gaussian * sigma
                tmp_index += count

            wgt[:] = 1.0
            self.swarm.restoreField(cellid_name)
            self.swarm.restoreField("velocity")
            self.swarm.restoreField("weight")

        #Change value for graphs
        self.xlim = 8.0
        self.ylim = 8.0

    def _construct_grid(self):

        grid_x = np.linspace(-self.xlim, self.xlim, self.bins + 1)
        grid_y = np.linspace(-self.ylim, self.ylim, self.bins + 1)
        self.grid_x = grid_x
        self.grid_y = grid_y


    def diagnostics(self, step=0):
        vel = self.swarm.getField("velocity")
        try:
            V = vel.reshape(self.nlocal, self.dim)

            local_n = self.nlocal
            global_n = self.comm.allreduce(local_n, op=MPI.SUM)

            local_mom = V.sum(axis=0)
            global_mom = self.comm.allreduce(local_mom, op=MPI.SUM)

            local_energy = 0.5 * self.mass * np.sum(V * V)
            global_energy = self.comm.allreduce(local_energy, op=MPI.SUM)

            mean_u = global_mom / global_n
            temp = (2.0 / self.dim) * global_energy / (global_n * self.mass)
            
            self.history["step"].append(step)
            self.history["temperature"].append(temp)
            self.history["energy"].append(global_energy)
            self.history["momentum_1"].append(np.linalg.norm(mean_u[0]))
            self.history["momentum_2"].append(np.linalg.norm(mean_u[1]))

            return {
                "N": global_n,
                "mean_u": mean_u,
                "energy": global_energy,
                "temperature": temp,
                "momentum_1": mean_u[0],
            }
        finally:
            self.swarm.restoreField("velocity")

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

    def bgk_collision_step(self):
        cells = self._get_particle_cells()
        cell_lists = _build_cell_lists(cells)
        vel = self.swarm.getField("velocity")
        for cell, plist in cell_lists.items():
            count = len(plist)
            gaussian = self.rng.normal(size=(count, self.dim))
            v_exact = np.sum(vel[plist], axis=0)/count
            e_exact = np.sum(vel[plist]**2, axis=0)/count

            #Smoothing
            v_prime = np.sum(gaussian, axis=0)/count
            e_prime = np.sum(gaussian**2, axis=0)/count
            tau = np.sqrt((e_prime-v_prime**2)/(e_exact-v_exact**2))
            lam = v_prime - tau* v_exact
            gaussian = (gaussian - lam)/tau
            #Smoothing twice for better results
            v_prime = np.sum(gaussian, axis=0)/count
            e_prime = np.sum(gaussian**2, axis=0)/count
            tau = np.sqrt((e_prime-v_prime**2)/(e_exact-v_exact**2))
            lam = v_prime - tau* v_exact
            gaussian = (gaussian - lam)/tau
            if count == 0:
                continue
            vel[plist, :] = gaussian 
        self.swarm.restoreField("velocity")

    def nanbu_collision_step(self):
        cells = self._get_particle_cells()
        cell_lists = _build_cell_lists(self._get_particle_cells())
        vel = self.swarm.getField("velocity")
        for cell, plist in cell_lists.items():
            number_points = len(plist)
            if number_points < 2:
                pass
            else:
                # Number of collision trials on this rank
                Mcol = int(0.5 * self.nu * number_points * self.dt)
                Mcol = Mcol if Mcol%2 == 0 else Mcol - 1  # ensure even number for pairing
                if Mcol == 0:
                    continue
                if 2*Mcol > number_points:
                    Mcol = Mcol - 2

                # Random particle pairs (allowing repeated use within a timestep)
                rnd_pairs = np.random.choice(plist, 2*Mcol, replace=False)
                i = rnd_pairs[:Mcol]
                j = rnd_pairs[Mcol:2*Mcol]
                
                vi = vel[i].copy()
                vj = vel[j].copy()

                g = vi - vj
                G = 0.5 * (vi + vj)

                # Isotropic scattering direction
                sigma = self._sample_angle(Mcol)

                # For elastic equal-mass collisions:
                # post-collisional relative velocity has same magnitude, random direction
                g_norm = np.linalg.norm(g, axis=1)

                vel[i, 0] = G[:, 0] + 0.5 * g_norm * np.cos(sigma)[:, 0]
                vel[i, 1] = G[:, 1] + 0.5 * g_norm * np.sin(sigma)[:, 0]
                vel[j, 0] = G[:, 0] - 0.5 * g_norm * np.cos(sigma)[:, 0]
                vel[j, 1] = G[:, 1] - 0.5 * g_norm * np.sin(sigma)[:, 0]
                del vi, vj, rnd_pairs, i, j, g, G, g_norm, Mcol, sigma
            del plist, number_points
        self.swarm.restoreField("velocity")
        del vel

    def _reflect_1d(self, x, v, xmin, xmax):
        """
        Reflect positions/velocities on [xmin, xmax].

        Works even if a particle crosses a wall more than once in one timestep.
        x, v are 1D numpy arrays (views).
        """
        L = xmax - xmin
        I_left = np.where(x < xmin)
        I_right = np.where(x > xmax)
        x[I_left] = -np.abs(x[I_left])*np.sign(v[I_left])
        x[I_right] = L - np.abs(x[I_right]-xmax)*np.sign(v[I_right])
        v[I_left] = -v[I_left]
        v[I_right] = -v[I_right]

    def transport_step(self, dt):
        self.swarm.sortGetAccess()
        celldm = self.swarm.getCellDMActive()
        coord_names = celldm.getCoordinateFields()
        pos = self.swarm.getField(coord_names[0])
        vel = self.swarm.getField("velocity")
        X = pos.reshape(self.nlocal, self.mesh_dim)
        V = vel.reshape(self.nlocal, self.dim)

        for d in range(self.effective_dim):
            X[:, d] += V[:, d] * dt
        self._reflect_1d(X[:, 0], V[:, 0], 0.0, self.info["Lx"])

        self.swarm.restoreField(coord_names[0])
        self.swarm.restoreField("velocity")
        self.swarm.migrate(remove_sent_points=False)
        self.swarm.sortRestoreAccess()

    def plot_velocity_histograms(self, prefix=""):
        vel = self.swarm.getField("velocity")
        try:
            Vlocal = vel.reshape(self.nlocal, self.dim).copy()
        finally:
            self.swarm.restoreField("velocity")

        gathered = self.comm.gather(Vlocal, root=0)
        if self.rank != 0:
            return

        V = np.vstack(gathered)

        fig, ax = plt.subplots(figsize=(7, 4))
        H, xedges, yedges = np.histogram2d(V[:,0], V[:,1], bins=(self.grid_x, self.grid_y))

        H = H.T
        plt.imshow(H, interpolation='nearest', origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        ax.set_xlim(-self.xlim, self.xlim)
        ax.set_ylim(-self.ylim, self.ylim)
        fig.savefig(f"{prefix}_vel.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


    
    def plot_observables(self, prefix=""):
        celldm = self.swarm.getCellDMActive()
        cellid_name = celldm.getCellID()
        coord_names = celldm.getCoordinateFields()
        cellid = self.swarm.getField(cellid_name)
        X = self.swarm.getField(coord_names[0])
        vel = self.swarm.getField("velocity")

        counts, bins = np.histogram(X[:,0], bins=self.bins)
        rho_x = (counts / self.N)*(self.bins/self.info["Lx"])*self.info["norm_rho"]
        vel_x = np.bincount(cellid[:,0], weights=vel[:,0]) / counts
        T = self.mass*(vel[:,0]**2 + vel[:,1]**2) 
        temp_x = np.bincount(cellid[:,0], weights=T) / counts - self.mass*vel_x**2
        
        self.swarm.restoreField(cellid_name)
        self.swarm.restoreField(coord_names[0])
        self.swarm.restoreField("velocity")

        fig, ax = plt.subplots(figsize=(7, 4))
        plt.plot(self.edges_x[:-1], rho_x)
        ax.set_xlabel("x")
        ax.set_ylabel(r"$\rho$")
        ax.set_title("Density profile")
        fig.tight_layout()
        fig.savefig(f"{prefix}_density_x.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        plt.plot(self.edges_x[:-1], vel_x)
        ax.set_xlabel("x")
        ax.set_ylabel(r"$u_x$")
        ax.set_title("Mean velocity X component profile")
        fig.tight_layout()
        fig.savefig(f"{prefix}_velocity_x.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        plt.plot(self.edges_x[:-1], temp_x)
        ax.set_xlabel("x")
        ax.set_ylabel(r"$T$")
        ax.set_title("Temperature profile")
        fig.tight_layout()
        fig.savefig(f"{prefix}_temperature_x.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        del fig, ax

    def run(self, nsteps: int, monitor_every: int = 10):
        if self.rank == 0:
            d = self.diagnostics()
            print(
                f"[step 0] N={d['N']}, t = 0.0 "
                f"T={d['temperature']:.6e} "
                f"|u_x|={np.linalg.norm(d['momentum_1']):.6e} "
                f"|u|={np.linalg.norm(d['mean_u']):.6e} "
                f"E={d['energy']:.6e}"
            )
        self._construct_grid()
        if self.rank == 0:
            self.plot_observables(prefix=f"{self.output_path}/dsmc_0_")
            self.plot_velocity_histograms(prefix=f"{self.output_path}/dsmc_0_")
        for step in range(1, nsteps + 1):
            self.transport_step(dt=0.5*self.dt)
            for coll_index in range(self.extra_collision):
                if self.collision_type == "nanbu":
                    self.nanbu_collision_step()
                elif self.collision_type == "bgk":
                    self.bgk_collision_step()
                else:
                    raise ValueError(f"Unknown collision type: {self.collision_type}")
            self.transport_step(dt=0.5*self.dt)
            d = self.diagnostics(step=step)
            if self.rank == 0:
                print(
                    f"[step {step}] N={d['N']}, t ={step*self.dt:.6f} "
                    f"T={d['temperature']:.6e} "
                    f"|u_x|={np.linalg.norm(d['momentum_1']):.6e} "
                    f"|u|={np.linalg.norm(d['mean_u']):.6e} "
                    f"E={d['energy']:.6e}"
                )
                if step % monitor_every == 0 or step == nsteps:
                    self.plot_observables(prefix=f"{self.output_path}/dsmc_{step}")
                    self.plot_velocity_histograms(prefix=f"{self.output_path}/dsmc_{step}")
            gc.collect()
                


def main():
    Opt = PETSc.Options()
    Print("Running homogeneous Maxwell DSMC with options:")

    nlocal = Opt.getReal("nlocal", 20000)
    nlocal = int(nlocal)
    nu = Opt.getReal("nu", 1.0)
    bins = Opt.getInt("bins", 31)
    dt = Opt.getReal("dt", 1e-2)
    nsteps = Opt.getInt("nsteps", 200)
    temp = Opt.getReal("temperature", 1.0)
    mass = Opt.getReal("mass", 1.0)
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
    Print(f"  temperature={temp}")
    Print(f"  mass={mass}")
    Print(f"  seed={seed}")
    Print(f"  monitor_every={monitor_every}")
    Print(f"  extra_collision={extra_collision}")
    Print(f"  collision_type={collision_type}")
    Print(f"  grazing_collision={grazing_collision}")

    Print("--------------------------------------------------------------------")

    sim = MaxwellDSMC(
        nlocal=nlocal,
        nu=nu,
        dt=dt,
        bins = bins,
        temperature=temp,
        mass=mass,
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
