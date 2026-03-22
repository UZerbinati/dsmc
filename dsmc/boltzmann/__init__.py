import petsc4py
import numpy as np
import sys
import os
petsc4py.init(sys.argv)
from petsc4py import PETSc
from mpi4py import MPI
import gc
import pickle


class BoltzmannDSMC:
    """
    DSMC for the space-inhomogeneous Boltzmann equation with Maxwell molecules.

    Particles carry velocity v in R^2 and move in a 1D spatial domain [0, Lx].
    Per-cell Nanbu or BGK collision steps. Strang splitting in time.
    Reflective boundary conditions.
    """

    def __init__(
        self,
        opts: dict,
        info: dict = {},
        comm: MPI.Comm = MPI.COMM_WORLD,
    ):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

        self.dim = 2
        self.nlocal = int(opts["nlocal"])
        self.N = self.nlocal * self.size
        self.nu = opts.get("nu", 1.0)
        self.dt = opts.get("dt", 1e-2)
        if 1 / self.nu < self.dt:
            raise RuntimeError("Time step too large for the collision frequency specified.")
        self.temperature = info.get("temperature", 1.0)
        self.mass = info.get("mass", 1.0)
        self.bins = opts.get("bins", 31)
        self.test = opts.get("test", "sod")
        self.prefix = opts.get("prefix", "")
        self.extra_collision = opts.get("extra_collision", 1)
        self.collision_type = opts.get("collision_type", "nanbu")
        self.effective_dim = 1 if self.test == "sod" else self.dim

        self.xlim = 10.0
        self.ylim = 10.0

        self.rng = np.random.default_rng(opts.get("seed", 1234) + self.rank)
        self.info = dict(info)

        self.history = {
            "step": [],
            "temperature": [],
            "energy": [],
            "momentum_1": [],
            "momentum_2": [],
        }

        self.output_path = f'{self.prefix}_output_boltzmann_{self.collision_type}'
        if self.rank == 0:
            os.makedirs(self.output_path, exist_ok=True)
        self.comm.Barrier()

        self.dm = self._create_mesh()
        self.mesh_dim = self.dm.getDimension()
        self.swarm = self._create_swarm()

        from dsmc.plot import (init_plot, plot_observables,
                               plot_cylinder_flow_observables,
                               plot_velocity_histograms, plot_history)
        from .transport import transport_step
        from .collision import nanbu_collision_step, bgk_collision_step
        from .initial import initialize_particles

        self.initialize_particles = initialize_particles.__get__(self)
        if self.test == "cylinder_flow":
            self.plot_observables = plot_cylinder_flow_observables.__get__(self)
        else:
            self.plot_observables = plot_observables.__get__(self)
        self.plot_velocity_histograms = plot_velocity_histograms.__get__(self)
        self.plot_history = plot_history.__get__(self)
        self.transport_step = transport_step.__get__(self)
        self.nanbu_collision_step = nanbu_collision_step.__get__(self)
        self.bgk_collision_step = bgk_collision_step.__get__(self)

        self.initialize_particles()
        init_plot()

    def _create_mesh(self):
        if self.test == "sod":
            Lx = 1.0
            self.info["Lx"] = Lx
            nx = self.bins
            self.edges_x = np.linspace(0.0, Lx, nx + 1)
            dm = PETSc.DMDA().create([nx + 1, 2], dof=1, stencil_width=1, comm=self.comm)
            dm.setUp()
            dm.setUniformCoordinates(0.0, Lx, 0.0, 1.0)
        elif self.test == "cylinder_flow":
            xmin = self.info.get("xmin", -8.0)
            xmax = self.info.get("xmax", 12.0)
            ymin = self.info.get("ymin", -5.0)
            ymax = self.info.get("ymax",  5.0)
            self.info["xmin"] = xmin
            self.info["xmax"] = xmax
            self.info["ymin"] = ymin
            self.info["ymax"] = ymax
            nx = self.bins
            ny = self.bins
            self.edges_x = np.linspace(xmin, xmax, nx + 1)
            self.edges_y = np.linspace(ymin, ymax, ny + 1)
            dm = PETSc.DMDA().create([nx + 1, ny + 1], dof=1, stencil_width=1, comm=self.comm)
            dm.setUp()
            dm.setUniformCoordinates(xmin, xmax, ymin, ymax)
        else:
            raise ValueError(f"Unknown test: {self.test}")
        return dm

    def _create_swarm(self):
        swarm = PETSc.DMSwarm().create(comm=self.comm)
        swarm.setDimension(self.mesh_dim)
        swarm.setType(PETSc.DMSwarm.Type.PIC)
        swarm.setCellDM(self.dm)

        swarm.initializeFieldRegister()
        swarm.registerField("velocity", self.dim, dtype=PETSc.RealType)
        swarm.registerField("weight", 1, dtype=PETSc.RealType)
        swarm.finalizeFieldRegister()

        # Buffer must be large enough to absorb incoming particles during migration.
        # In the worst case (e.g. Sod shock tube) one rank may receive up to N
        # particles, so we use N as the buffer.
        swarm.setLocalSizes(self.nlocal, self.N)
        return swarm

    def _construct_grid(self):
        self.grid_x = np.linspace(-self.xlim, self.xlim, self.bins + 1)
        self.grid_y = np.linspace(-self.ylim, self.ylim, self.bins + 1)
        self.delta_x = 2 * self.xlim / self.bins
        self.delta_y = 2 * self.ylim / self.bins

    def diagnostics(self, step=0):
        vel = self.swarm.getField("velocity")
        V = vel.reshape(self.nlocal, self.dim)

        local_n = self.nlocal
        global_n = self.comm.allreduce(local_n, op=MPI.SUM)

        local_mom = V.sum(axis=0)
        global_mom = self.comm.allreduce(local_mom, op=MPI.SUM)

        local_energy = 0.5 * self.mass * np.sum(V * V)
        global_energy = self.comm.allreduce(local_energy, op=MPI.SUM)

        self.swarm.restoreField("velocity")

        mean_u = global_mom / global_n
        temp = (2.0 / self.dim) * global_energy / (global_n * self.mass)

        self.history["step"].append(step)
        self.history["temperature"].append(temp)
        self.history["energy"].append(global_energy)
        self.history["momentum_1"].append(float(mean_u[0]))
        self.history["momentum_2"].append(float(mean_u[1]))

        if self.rank == 0:
            with open(f'{self.output_path}/history.pickle', 'wb') as fp:
                pickle.dump(self.history, fp)

        return {
            "N": global_n,
            "mean_u": mean_u,
            "energy": global_energy,
            "temperature": temp,
        }

    def run(self, nsteps: int, monitor_every: int = 10):
        d = self.diagnostics()
        if self.rank == 0:
            print(
                f"[step 0] N={d['N']}, t=0.0 "
                f"T={d['temperature']:.6e} "
                f"|u|={np.linalg.norm(d['mean_u']):.6e} "
                f"E={d['energy']:.6e}"
            )
        self._construct_grid()
        self.plot_observables(prefix=f"{self.output_path}/dsmc_0")
        self.plot_velocity_histograms(prefix=f"{self.output_path}/dsmc_0")
        for step in range(1, nsteps + 1):
            self.transport_step(dt=0.5 * self.dt)
            for _ in range(self.extra_collision):
                if self.collision_type == "nanbu":
                    self.nanbu_collision_step()
                elif self.collision_type == "bgk":
                    self.bgk_collision_step()
                else:
                    raise ValueError(f"Unknown collision type: {self.collision_type}")
            self.transport_step(dt=0.5 * self.dt)
            d = self.diagnostics(step=step)
            if self.rank == 0:
                print(
                    f"[step {step}] N={d['N']}, t={step*self.dt:.6f} "
                    f"T={d['temperature']:.6e} "
                    f"|u|={np.linalg.norm(d['mean_u']):.6e} "
                    f"E={d['energy']:.6e}"
                )
            if step % monitor_every == 0 or step == nsteps:
                self.plot_observables(prefix=f"{self.output_path}/dsmc_{step}")
                self.plot_velocity_histograms(prefix=f"{self.output_path}/dsmc_{step}")
                if self.rank == 0:
                    self.plot_history(prefix=f"{self.output_path}/dsmc")
            gc.collect()
