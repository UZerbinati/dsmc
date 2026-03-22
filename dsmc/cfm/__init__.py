import petsc4py
import numpy as np
import sys
import os
petsc4py.init(sys.argv)
from petsc4py import PETSc
from mpi4py import MPI
import gc
import pickle

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
        vlasov_force = None,
        seed: int = 1234,
        bins: int = 31,
        test: str = "sod",
        variance: str = "circle",
        prefix: str = "",
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
        if 1/nu < dt:
            raise RuntimeError("You have too large of a time-step for the collisional frequency you specified")
        self.info = info
        self.bins = bins
        self.delta_bins = 1/(bins+1)
        self.test = test
        self.variance = variance
        self.prefix = prefix
        self.extra_collision = extra_collision
        self.grazing_collision = grazing_collision
        self.collision_type = collision_type
        self.vlasov_force = vlasov_force
        self.dump = "hist"


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
            "circular_var": []
        }

        self.output_path = f'{self.prefix}_output_cfm_{self.collision_type}' 
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        self.dm = self._create_mesh()
        self.mesh_dim = self.dm.getDimension()
        self.swarm = self._create_swarm()
        
        from .plot import init_plot, plot_histograms, plot_history
        from .transport import transport_step
        from .collision import nanbu_collision_step
        from .initial import initialize_particles

        self.initialize_particles = initialize_particles.__get__(self)
        self.plot_histograms = plot_histograms.__get__(self)
        self.plot_history = plot_history.__get__(self)
        self.transport_step = transport_step.__get__(self)
        self.nanbu_collision_step = nanbu_collision_step.__get__(self)

        self.initialize_particles()
        init_plot() 
        

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
        
        #Circular stats
        #TODO: Think how to parallerlise this!
        if self.variance == "circle":
            z = np.exp(1j*angle)
        elif self.variance == "real_projective_plane":
            z = np.exp(2j*np.mod(angle, np.pi))
        else:
            raise RuntimeError(f"[!] Do not know how to compute the variance for {self.variance}")
        m = np.sum(z)/self.nlocal
        R = np.abs(m)

        
        self.history["step"].append(step)
        self.history["temperature"].append(temp)
        self.history["momentum_1"].append(np.linalg.norm(mean_u[0]))
        self.history["momentum_2"].append(np.linalg.norm(mean_u[1]))
        self.history["ang_momentum"].append(np.linalg.norm(mean_eta))
        self.history["circular_var"].append(1-R)

        self.swarm.restoreField("orientation")
        self.swarm.restoreField("velocity")
        self.swarm.restoreField("angular_velocity")


        with open(f'{self.output_path}/history.pickle', 'wb') as fp:
            pickle.dump(self.history, fp)

        return {
            "N": global_n,
            "mean_u": mean_u,
            "temperature": temp,
            "circular_var": 1-R
        }

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
        Maxwellian_y = np.sum(Maxwellian, axis=(0,2))*self.delta_x*self.delta_omega
        Maxwellian_omega = np.sum(Maxwellian, axis=(0,1))*self.delta_x*self.delta_y
        return Maxwellian, Maxwellian_x, Maxwellian_y, Maxwellian_omega

    def run(self, nsteps: int, monitor_every: int = 10):
        if self.rank == 0:
            d = self.diagnostics()
            print(
                f"[step 0] N={d['N']}, t = 0.0 "
                f"T={d['temperature']:.6e} "
                f"|u|={np.linalg.norm(d['mean_u']):.6e} "
                f"circ_var={np.linalg.norm(d['circular_var']):.6e} "
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
                    f"circ_var={np.linalg.norm(d['circular_var']):.6e} "
                )
                if step % monitor_every == 0 or step == nsteps:
                    self.plot_histograms(prefix=f"{self.output_path}/dsmc_{step}")
                    self.plot_history(prefix=f"{self.output_path}/dsmc")
            gc.collect()
