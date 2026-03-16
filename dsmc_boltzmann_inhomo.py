#!/usr/bin/env python3
import petsc4py
import numpy as np
import sys
#passing system argument to PETSc
petsc4py.init(sys.argv)
from petsc4py import PETSc
from mpi4py import MPI
import matplotlib.pyplot as plt


def Print(*args, **kwargs):
    """Helper to print only from rank 0."""
    if PETSc.COMM_WORLD.Get_rank() == 0:
        print(*args, **kwargs)

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
        self.nu = nu              # collision frequency
        self.nu=1/dt
        self.dt = dt
        self.temperature = temperature
        self.mass = mass
        self.bins = bins
        self.test = test


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
            vel = self.swarm.getField("velocity")
            wgt = self.swarm.getField("weight")
            X = np.zeros((self.nlocal, self.mesh_dim))
            V = vel.reshape(self.nlocal, self.dim)
            W = wgt.reshape(self.nlocal)
            n_left = int(0.89*self.nlocal)
            n_right = self.nlocal - n_left
            X[:n_left, 0] = self.rng.uniform(0.0, 0.5*self.info["Lx"], size=n_left)
            X[n_left:, 0] = self.rng.uniform(0.5, 1*self.info["Lx"], size=n_right)
            X[:,0] = np.sort(X[:,0]) 
            X[:,1] = 0.5
            counts, bins = np.histogram(X[:,0], bins=self.bins)
            # anisotropic Gaussian
            tmp_index = 0
            for i in range(self.bins):
                count = counts[i]
                if count == 0:
                    continue
                if i > self.bins//2:
                    V[tmp_index:tmp_index+count, :] = self.rng.normal(size=(count, self.dim))* np.sqrt(0.8)*sigma
                else:
                    V[tmp_index:tmp_index+count, :] = self.rng.normal(size=(count, self.dim)) * sigma
                tmp_index += count

            W[:] = 1.0
            self.swarm.restoreField("velocity")
            self.swarm.restoreField("weight")
            self.swarm.setPointCoordinates(X)
            self.swarm.migrate(remove_sent_points=True)

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
            }
        finally:
            self.swarm.restoreField("velocity")

    def _sample_angle(self, m: int):
        Theta = self.rng.uniform(size=(m, 1)) * 2 * np.pi
        return Theta

    def collision_step(self):
        """
        One homogeneous DSMC collision step.

        Expected number of binary collisions per rank:
            M ~ 0.5 * nu * N_local * dt

        For Maxwell molecules this is consistent with uniform pair selection,
        because the kernel is independent of relative speed magnitude.
        """
        ncells, _ = self.swarm.sortGetSizes()
        vel = self.swarm.getField("velocity")
        V = vel.reshape(self.nlocal, self.dim)
        for cell_id in range(ncells):
            plist = self.swarm.sortGetPointsPerCell(e)
            number_points = len(plist)
            V = V[plist]
            if m < 2:
                continue
            else:
                # Number of collision trials on this rank
                Mcol = int(0.5 * self.nu * number_points * self.dt)
                Mcol = Mcol if Mcol%2 == 0 else Mcol - 1  # ensure even number for pairing
                if Mcol == 0:
                    return
                if 2*Mcol > self.nlocal:
                    Mcol = Mcol - 2

                # Random particle pairs (allowing repeated use within a timestep)
                rnd_pairs = np.random.choice(plist, 2*Mcol, replace=False)
                i = rnd_pairs[:Mcol]
                j = rnd_pairs[Mcol:2*Mcol]
                
                vi = V[i].copy()
                vj = V[j].copy()

                g = vi - vj
                G = 0.5 * (vi + vj)

                # Isotropic scattering direction
                sigma = self._sample_angle(Mcol)

                # For elastic equal-mass collisions:
                # post-collisional relative velocity has same magnitude, random direction
                g_norm = np.linalg.norm(g, axis=1)

                V[i, 0] = G[:, 0] + 0.5 * g_norm * np.cos(sigma)[:, 0]
                V[i, 1] = G[:, 1] + 0.5 * g_norm * np.sin(sigma)[:, 0]
                V[j, 0] = G[:, 0] - 0.5 * g_norm * np.cos(sigma)[:, 0]
                V[j, 1] = G[:, 1] - 0.5 * g_norm * np.sin(sigma)[:, 0]
        self.swarm.restoreField("velocity")
        self.swarm.sortRestoreAccess()

    def transport_step(self, dt):
        celldm = self.swarm.getCellDMActive()
        coord_names = celldm.getCoordinateFields()
        pos = self.swarm.getField(coord_names[0])
        vel = self.swarm.getField("velocity")
        X = pos.reshape(self.nlocal, self.mesh_dim)
        V = vel.reshape(self.nlocal, self.dim)

        for d in range(self.mesh_dim):
            X[:, d] += V[:, d] * dt

        self.swarm.restoreField(coord_names[0])
        self.swarm.restoreField("velocity")
        self.swarm.migrate(remove_sent_points=False)

    
    def plot_observables(self, prefix=""):
        celldm = self.swarm.getCellDMActive()
        cellid_name = celldm.getCellID()
        coord_names = celldm.getCoordinateFields()
        cellid = self.swarm.getField(cellid_name)
        X = self.swarm.getField(coord_names[0])
        counts, bins = np.histogram(X[:,0], bins=self.bins)
        rho_x = (counts / self.N)*(self.bins/self.info["Lx"])*self.info["norm_rho"]
        vel = self.swarm.getField("velocity")
        V = vel.reshape(self.nlocal, self.dim)
        vel_x = np.bincount(cellid[:,0], weights=V[:,0]) / counts
        T = self.mass*(V[:,0]**2 + V[:,1]**2) 
        temp_x = np.bincount(cellid[:,0], weights=T) / counts - self.mass*vel_x**2
        self.swarm.restoreField(cellid_name)
        self.swarm.restoreField(coord_names[0])
        self.swarm.restoreField("velocity")
        self.swarm.sortRestoreAccess()

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



    def run(self, nsteps: int, monitor_every: int = 10):
        if self.rank == 0:
            d = self.diagnostics()
            print(
                f"[step 0] N={d['N']} "
                f"T={d['temperature']:.6e} "
                f"|u|={np.linalg.norm(d['mean_u']):.6e} "
                f"E={d['energy']:.6e}"
            )
        self._construct_grid()
        self.plot_observables(prefix=f"output/dsmc_0_")
        for step in range(1, nsteps + 1):
            self.transport_step(dt=0.5*self.dt)
            self.collision_step()
            self.transport_step(dt=0.5*self.dt)
            if step % monitor_every == 0 or step == nsteps:
                d = self.diagnostics(step=step)
                if self.rank == 0:
                    print(
                        f"[step {step}] N={d['N']} "
                        f"T={d['temperature']:.6e} "
                        f"|u|={np.linalg.norm(d['mean_u']):.6e} "
                        f"E={d['energy']:.6e}"
                    )
            self.plot_observables(prefix=f"output/dsmc_{step}_")
                


def main():
    Opt = PETSc.Options()
    Print("Running homogeneous Maxwell DSMC with options:")

    nlocal = Opt.getInt("nlocal", 20000)
    nu = Opt.getReal("nu", 1.0)
    bins = Opt.getInt("bins", 31)
    dt = Opt.getReal("dt", 1e-2)
    nsteps = Opt.getInt("nsteps", 200)
    temp = Opt.getReal("temperature", 1.0)
    mass = Opt.getReal("mass", 1.0)
    seed = Opt.getInt("seed", 1234)
    monitor_every = Opt.getInt("monitor_every", 10)

    Print(f"  nlocal={nlocal}")
    Print(f"  nu={nu}")
    Print(f"  dt={dt}")
    Print(f"  bins={bins}")
    Print(f"  nsteps={nsteps}")
    Print(f"  temperature={temp}")
    Print(f"  mass={mass}")
    Print(f"  seed={seed}")
    Print(f"  monitor_every={monitor_every}")

    Print("--------------------------------------------------------------------")

    sim = MaxwellDSMC(
        nlocal=nlocal,
        nu=nu,
        dt=dt,
        bins = bins,
        temperature=temp,
        mass=mass,
        seed=seed,
        comm=MPI.COMM_WORLD,
    )
    sim.run(nsteps=nsteps, monitor_every=monitor_every)
    Print("Simulation complete.")

if __name__ == "__main__":
    main()
