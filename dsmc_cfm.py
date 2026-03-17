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
        self.info = info
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
        vel[:] = self.rng.uniform(size=(self.nlocal, 2), low=-1, high=1)
        angular_vel[:] = self.rng.uniform(size=(self.nlocal, 1), low=-0.5, high=0.5)
        wgt[:] = 1.0
        self.swarm.restoreField("orientation")
        self.swarm.restoreField("velocity")
        self.swarm.restoreField("angular_velocity")
        self.swarm.restoreField("weight")

        #Change value for graphs
        self.xlim = 4.0
        self.ylim = 4.0

    def _construct_grid(self):

        grid_x = np.linspace(-self.xlim, self.xlim, self.bins + 1)
        grid_y = np.linspace(-self.ylim, self.ylim, self.bins + 1)
        self.grid_x = grid_x
        self.grid_y = grid_y


    def diagnostics(self, step=0):
        vel = self.swarm.getField("velocity")
        omega = self.swarm.getField("angular_velocity")

        local_n = self.nlocal
        global_n = self.comm.allreduce(local_n, op=MPI.SUM)

        local_mom = vel.sum(axis=0)
        global_mom = self.comm.allreduce(local_mom, op=MPI.SUM)

        local_ang_mom = self.info["inertia"]*omega.sum(axis=0) 
        global_ang_mom = self.comm.allreduce(local_ang_mom, op=MPI.SUM)

        local_energy = 0.5 * self.info["mass"] * np.sum(vel*vel) + 0.5*self.info["inertia"]*np.sum(omega*omega)
        global_energy = self.comm.allreduce(local_energy, op=MPI.SUM)

        mean_u = global_mom / global_n
        mean_eta = global_ang_mom/global_n
        temp = (2.0 / (self.dim+2)) * global_energy / global_n 
        
        self.history["step"].append(step)
        self.history["temperature"].append(temp)
        self.history["momentum_1"].append(np.linalg.norm(mean_u[0]))
        self.history["momentum_2"].append(np.linalg.norm(mean_u[1]))
        self.history["ang_momentum"].append(np.linalg.norm(mean_eta))
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
        vel = self.swarm.getField("velocity")
        # Number of collision trials on this rank
        Mcol = int(0.5 * self.nu * self.nlocal * self.dt)
        Mcol = Mcol if Mcol%2 == 0 else Mcol - 1  # ensure even number for pairing
            
        if 2*Mcol > self.nlocal:
            Mcol = Mcol - 2

        # Random particle pairs (allowing repeated use within a timestep)
        rnd_pairs = np.random.choice(range(0,self.nlocal), 2*Mcol, replace=False)
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
        self.swarm.restoreField("velocity")
    """
    def _reflect_1d(self, x, v, xmin, xmax):
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
    """
    def plot_velocity_histograms(self, prefix=""):
        vel = self.swarm.getField("velocity")
        Vlocal = vel.reshape(self.nlocal, self.dim).copy()
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
        if self.grazing_collision:
        fig.savefig(f"{prefix}_vel.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    

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
            self.plot_velocity_histograms(prefix=f"{self.output_path}/dsmc_0")
        for step in range(1, nsteps + 1):
            #self.transport_step(dt=0.5*self.dt)
            for coll_index in range(self.extra_collision):
                if self.collision_type == "nanbu":
                    self.nanbu_collision_step()
                else:
                    raise ValueError(f"Unknown collision type: {self.collision_type}")
            #self.transport_step(dt=0.5*self.dt)
            d = self.diagnostics(step=step)
            if self.rank == 0:
                print(
                    f"[step {step}] N={d['N']}, t ={step*self.dt:.6f} "
                    f"T={d['temperature']:.6e} "
                    f"|u|={np.linalg.norm(d['mean_u']):.6e} "
                )
                if step % monitor_every == 0 or step == nsteps:
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
            "length": 1.0}

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
