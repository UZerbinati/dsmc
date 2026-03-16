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

class HomogeneousMaxwellDSMC:
    """
    DSMC for space-homogeneous Maxwell molecules.

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
        bins: int = 60,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

        self.dim = 2
        self.nlocal = nlocal
        self.nu = nu              # collision frequency
        self.nu=1/dt
        self.dt = dt
        self.temperature = temperature
        self.mass = mass
        self.bins = bins


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

        self.swarm = self._create_swarm()
        self._initialize_particles()

    def _create_swarm(self):
        swarm = PETSc.DMSwarm().create(comm=self.comm)
        # For this use-case we only need DMSwarm as a generic particle container.
        # In most PETSc builds the BASIC type is appropriate here.
        swarm.setType(PETSc.DMSwarm.Type.BASIC)

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
        sigma = np.sqrt(self.temperature / self.mass)

        vel = self.swarm.getField("velocity")
        wgt = self.swarm.getField("weight")
        try:
            V = vel.reshape(self.nlocal, self.dim)
            W = wgt.reshape(self.nlocal)
        
            # anisotropic Gaussian
            sigmas = np.array([2.0, 0.7, 0.4][:self.dim])
            V[:] = self.rng.normal(size=(self.nlocal, self.dim)) * sigmas
            W[:] = 1.0
        finally:
            self.swarm.restoreField("velocity")
            self.swarm.restoreField("weight")


        #Change value for graphs
        self.xlim = 8.0
        self.ylim = 8.0
        
        self._remove_mean_velocity()

    def _remove_mean_velocity(self):
        vel = self.swarm.getField("velocity")
        try:
            V = vel.reshape(self.nlocal, self.dim)
            local_momentum = V.sum(axis=0)
            global_momentum = self.comm.allreduce(local_momentum, op=MPI.SUM)
            global_n = self.comm.allreduce(self.nlocal, op=MPI.SUM)
            mean_u = global_momentum / global_n
            V[:] -= mean_u
        finally:
            self.swarm.restoreField("velocity")

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
        vel = self.swarm.getField("velocity")
        try:
            V = vel.reshape(self.nlocal, self.dim)

            if self.nlocal < 2:
                return

            # Number of collision trials on this rank
            Mcol = int(0.5 * self.nu * self.nlocal * self.dt)
            Mcol = Mcol if Mcol%2 == 0 else Mcol - 1  # ensure even number for pairing
            if Mcol == 0:
                return
            if 2*Mcol > self.nlocal:
                Mcol = Mcol - 2

            # Random particle pairs (allowing repeated use within a timestep)
            rnd_pairs = np.random.choice(range(0,self.nlocal), 2*Mcol, replace=False)
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

        finally:
            self.swarm.restoreField("velocity")

    def plot_history(self, prefix="output/dsmc_history"):
        if self.rank != 0:
            return

        steps = np.array(self.history["step"])
        temperature = np.array(self.history["temperature"])
        energy = np.array(self.history["energy"])
        momentum_1 = np.array(self.history["momentum_1"])
        momentum_2 = np.array(self.history["momentum_2"])

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(steps, temperature)
        ax.set_xlabel("step")
        ax.set_ylabel("temperature")
        ax.set_title("Temperature history")
        fig.tight_layout()
        fig.savefig(f"{prefix}_temperature.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(steps, energy)
        ax.set_xlabel("step")
        ax.set_ylabel("total kinetic energy")
        ax.set_title("Energy history")
        fig.tight_layout()
        fig.savefig(f"{prefix}_energy.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(steps, momentum_1)
        ax.set_xlabel("step")
        ax.set_ylabel(r"$u_x$")
        ax.set_title("Mean velocity X component history")
        fig.tight_layout()
        fig.savefig(f"{prefix}_momentum_x.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(steps, momentum_2)
        ax.set_xlabel("step")
        ax.set_ylabel(r"$u_y$")
        ax.set_title("Mean velocity Y component history")
        fig.tight_layout()
        fig.savefig(f"{prefix}_momentum_y.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    
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
        self.plot_velocity_histograms(prefix=f"output/dsmc_0_")
        for step in range(1, nsteps + 1):
            self.collision_step()

            if step % monitor_every == 0 or step == nsteps:
                d = self.diagnostics(step=step)
                if self.rank == 0:
                    print(
                        f"[step {step}] N={d['N']} "
                        f"T={d['temperature']:.6e} "
                        f"|u|={np.linalg.norm(d['mean_u']):.6e} "
                        f"E={d['energy']:.6e}"
                    )
                self.plot_velocity_histograms(prefix=f"output/dsmc_{step}_")
                


def main():
    Opt = PETSc.Options()
    Print("Running homogeneous Maxwell DSMC with options:")

    nlocal = Opt.getInt("nlocal", 20000)
    nu = Opt.getReal("nu", 1.0)
    dt = Opt.getReal("dt", 1e-2)
    nsteps = Opt.getInt("nsteps", 200)
    temp = Opt.getReal("temperature", 1.0)
    mass = Opt.getReal("mass", 1.0)
    seed = Opt.getInt("seed", 1234)
    monitor_every = Opt.getInt("monitor_every", 10)

    Print(f"  nlocal={nlocal}")
    Print(f"  nu={nu}")
    Print(f"  dt={dt}")
    Print(f"  nsteps={nsteps}")
    Print(f"  temperature={temp}")
    Print(f"  mass={mass}")
    Print(f"  seed={seed}")
    Print(f"  monitor_every={monitor_every}")

    Print("--------------------------------------------------------------------")

    sim = HomogeneousMaxwellDSMC(
        nlocal=nlocal,
        nu=nu,
        dt=dt,
        temperature=temp,
        mass=mass,
        seed=seed,
        comm=MPI.COMM_WORLD,
    )
    sim.run(nsteps=nsteps, monitor_every=monitor_every)
    Print("Simulation complete.")
    sim.plot_history()
    Print("Post-processing complete.")


if __name__ == "__main__":
    main()
