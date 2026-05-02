import petsc4py
import numpy as np
import sys
import os
petsc4py.init(sys.argv)
from petsc4py import PETSc
from mpi4py import MPI
import gc
import pickle

class CFMZNeedleDSMCHomo:
    """
    DSMC solver for the homogeneous CFMZ (Carrillo–Farrell–Medaglia–Zerbinati)
    kinetic equation for needle-like (oriented rigid rod) systems.

    Each particle carries:
      - translational velocity **v** ∈ R²
      - orientation θ ∈ (0, 2π)
      - angular velocity ω ∈ R

    Time stepping uses Strang splitting:
      [DKD](dt/2) → collision(s) → [DKD](dt/2)
    i.e. D(dt/4)·K(dt/2)·D(dt/4) → collision → D(dt/4)·K(dt/2)·D(dt/4),
    which is second-order symplectic for the combined Vlasov+collision system.

    The transport substep advances θ by ω·dt (plus an optional mean-field
    Vlasov torque) and wraps the angle back onto (0, 2π).  The collision
    substep performs rigid-rod impulse collisions: a random impact angle ψ
    and random contact arm ℓ are drawn; pairs that are nearly parallel
    fall back to a spherical-like collision.

    Parameters
    ----------
    opts : dict
        Simulation options.  Recognised keys:

        - ``nlocal``          (int)   particles per MPI rank.
        - ``nu``              (float) collision frequency; default 1.0.
        - ``dt``              (float) time step; must satisfy dt ≤ 1/nu.
        - ``bins``            (int)   histogram bins per axis; default 31.
        - ``test``            (str)   initial condition: ``"uniform_angle"``
                                      or ``"perturbed_uniform_angle"``.
        - ``collision_type``  (str)   only ``"nanbu"`` is implemented.
        - ``nu``              (float) for ``cross_section="maxwell"`` this is
                                      the exact collision frequency; the
                                      constraint dt ≤ 1/ν is enforced.  For
                                      ``"hard_needle"`` it is only the initial
                                      estimate of the NTC running maximum
                                      ``_nu_max``; a good choice is
                                      ``nu ≈ L · v_max`` where ``v_max`` is
                                      the expected maximum relative speed.
        - ``extra_collision`` (int)   collision sub-steps per time step; default 1.
        - ``variance``        (str)   circular variance geometry: ``"circle"``
                                      or ``"real_projective_plane"``; default ``"circle"``.
        - ``seed``            (int)   RNG seed; default 1234.
        - ``prefix``          (str)   path prefix for output directories.
        - ``transport``       (bool)  if ``False`` the transport substep is
                                      skipped entirely (pure collision); default ``True``.
        - ``T_bath``          (float or None)  target temperature for the Andersen
                                      thermostat.  ``None`` (default) disables the
                                      thermostat entirely (microcanonical run).
        - ``nu_bath``         (float) Andersen bath collision frequency; default 1.0.
                                      Each particle is resampled from the Maxwellian
                                      at ``T_bath`` with probability ``nu_bath·dt``
                                      per step.  Set equal to ``nu`` for coupling
                                      comparable to the physical collision rate.
        - ``init_at_T_bath`` (bool)  If ``True`` (default) and ``T_bath`` is set,
                                      initialise velocities from the Maxwellian at
                                      ``T_bath`` so the simulation starts at the
                                      target temperature.  Set to ``False`` to use
                                      the default uniform IC regardless of ``T_bath``.

    info : dict
        Physical parameters.  Required keys:

        - ``mass``     (float) translational mass m.
        - ``inertia``  (float) moment of inertia I.
        - ``length``   (float) rod half-length L.

        Optional keys:

        - ``cutoff``        (float) angular cutoff for near-parallel detection;
                                    default 0.1.
        - ``ev``            (float) translational restitution; default 1.0.
        - ``om``            (float) rotational restitution; default 1.0.
        - ``cross_section`` (str)   collision kernel.  ``"maxwell"`` (default)
                                    uses a uniform kernel — all pairs equally
                                    likely, matching the classical Nanbu method.
                                    ``"hard_needle"`` uses the Onsager
                                    excluded-volume kernel
                                    W = |g·n| · L|sin(θ₁−θ₂)| derived for
                                    2-D calamitic needles in Example B of
                                    arXiv:2508.10744; Bird's NTC
                                    acceptance–rejection is applied each step
                                    and the running maximum ``_nu_max`` adapts
                                    automatically.

    vlasov_force : callable or None
        If provided, called as ``vlasov_force(angle)`` each transport
        substep to add a mean-field torque to ω.  For the Onsager
        potential W(θ₁,θ₂) = |sin(θ₁−θ₂)| the torque is
        F(θ) = −L² ∫ sign(sin(θ−θ')) cos(θ−θ') ρ(θ') dθ'.

    interaction_energy : callable or None
        If provided, called as ``interaction_energy(angle)`` once per
        time step inside ``diagnostics`` to compute the mean-field
        interaction energy
        E[ρ] = ∫∫ W(θ₁,θ₂) ρ(θ₁) ρ(θ₂) dθ₁ dθ₂.
        The callable must handle its own MPI allreduce and return a
        global scalar.  When set, ``history["interaction_energy"]`` and
        ``history["total_energy"]`` (= kinetic + E[ρ]) are recorded and
        plotted alongside the other time-history quantities.

    comm : MPI.Comm
        MPI communicator; default ``MPI.COMM_WORLD``.
    """

    def __init__(
        self,
        opts: dict,
        info: dict = {},
        vlasov_force=None,
        interaction_energy=None,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

        self.dim = 2
        self.nlocal = int(opts["nlocal"])
        self.N = self.nlocal * self.size
        self.nu   = opts.get("nu", 1.0)
        self.dt   = opts.get("dt", 1e-2)
        self.info = info
        # For Maxwell molecules the dt ≤ 1/ν constraint must hold exactly.
        # For hard_needle the NTC running maximum _nu_max handles the rate
        # automatically, so the constraint is skipped (nu is only an initial
        # estimate and _nu_max may exceed it).
        if info.get("cross_section", "maxwell") == "maxwell" and 1 / self.nu < self.dt:
            raise RuntimeError("You have too large of a time-step for the collisional frequency you specified")
        # Running maximum collision rate for the NTC method (hard_needle).
        # Initialised to nu; grows monotonically during the simulation.
        self._nu_max = self.nu
        self.bins = opts.get("bins", 31)
        self.delta_bins = 1 / (self.bins + 1)
        self.test = opts.get("test", "uniform_angle")
        self.variance = opts.get("variance", "circle")
        self.n_modes = [int(n) for n in opts.get("n_modes", [2])]
        self.prefix = opts.get("prefix", "")
        self.extra_collision = opts.get("extra_collision", 1)
        self.grazing_collision = opts.get("grazing_collision", False)
        self.collision_type = opts.get("collision_type", "nanbu")
        self.vlasov_force = vlasov_force
        self.interaction_energy = interaction_energy
        self.transport = opts.get("transport", True)
        self.T_bath       = opts.get("T_bath",       None)
        self.nu_bath      = opts.get("nu_bath",      1.0)
        self.init_at_T_bath = opts.get("init_at_T_bath", True)
        self.dump = "hist"


        self.xlim = 10.0
        self.ylim = 10.0

        self.rng = np.random.default_rng(opts.get("seed", 1234) + self.rank)

        # History of simulation
        self.history = {
            "step": [],
            "temperature": [],
            "energy": [],
            "momentum_1": [],
            "momentum_2": [],
            "ang_momentum": [],
            "circular_var": [],
        }
        for n in self.n_modes:
            self.history[f"circular_var_n{n}"] = []
        if interaction_energy is not None:
            self.history["interaction_energy"] = []
            self.history["total_energy"] = []
            self.history["total_energy_rot"] = []

        self.output_path = f'{self.prefix}_output_cfmz_{self.collision_type}'
        if self.rank == 0:
            os.makedirs(self.output_path, exist_ok=True)
        self.comm.Barrier()

        self.dm = self._create_mesh()
        self.mesh_dim = self.dm.getDimension()
        self.swarm = self._create_swarm()
        
        from dsmc.plot import init_plot, plot_histograms, plot_history
        from .transport import transport_step, vlasov_kick_step
        from .collision import nanbu_collision_step, andersen_thermostat_step
        from .initial import initialize_particles

        self.initialize_particles = initialize_particles.__get__(self)
        self.plot_histograms = plot_histograms.__get__(self)
        self.plot_history = plot_history.__get__(self)
        self.transport_step = transport_step.__get__(self)
        self.vlasov_kick_step = vlasov_kick_step.__get__(self)
        self.nanbu_collision_step = nanbu_collision_step.__get__(self)
        self.andersen_thermostat_step = andersen_thermostat_step.__get__(self)

        self.initialize_particles()
        init_plot() 
        

    def _create_mesh(self):
        """Create a 1-D periodic DMDA over [0, 2π] (orientation space)."""
        nx = self.bins
        self.edges_x = np.linspace(0.0, 2*np.pi, nx + 1)
        dm = PETSc.DMDA().create([nx+1, 2], dof=1, stencil_width=1, comm=self.comm)
        dm.setUp()
        dm.setUniformCoordinates(0.0, 2*np.pi, 0.0, 1.0)
        return dm

    def _create_swarm(self):
        """Create the DMSwarm and register particle fields (orientation, velocity, angular_velocity, weight)."""
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
        """Build histogram grid edges for velocity (vx, vy) and angular (θ, ω) spaces."""
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
        """Compute and record global moments (called by all ranks).

        Validates that all orientations remain in (0, 2π), then computes
        total kinetic + rotational energy, mean translational and angular
        momenta, temperature, and circular variance of θ via MPI allreduce.
        Results are appended to ``self.history`` and serialised to disk by
        rank 0.

        Parameters
        ----------
        step : int
            Current time-step index.

        Returns
        -------
        dict with keys ``N``, ``mean_u``, ``temperature``, ``circular_var``.
        """
        angle = self.swarm.getField("orientation")
        vel = self.swarm.getField("velocity")
        omega = self.swarm.getField("angular_velocity")

        if sum(np.where(angle >= 2*np.pi,1,0))>0:
            raise RuntimeError("[!] Not sticking to the manifold!")
        if sum(np.where(angle <= 0, 1,0))>0:
            raise RuntimeError("[!] Not sticking to the manifold!")

        local_n        = self.nlocal
        local_mom      = vel.sum(axis=0)      # shape (dim,)
        local_ang_mom  = omega.sum(axis=0)    # shape (1,)
        local_energy_rot = 0.5 * self.info["inertia"] * np.sum(omega * omega)
        local_energy = (0.5 * self.info["mass"] * np.sum(vel * vel) +
                        local_energy_rot)

        # Legacy circular variance — drives ``history["circular_var"]``.
        # ``"circle"``                ⇒ R₁ = |⟨e^{iθ}⟩|     (polar)
        # ``"real_projective_plane"`` ⇒ R₂ = |⟨e^{2iθ}⟩|    (nematic)
        if self.variance == "circle":
            legacy_n = 1
        elif self.variance == "real_projective_plane":
            legacy_n = 2
        else:
            raise RuntimeError(f"[!] Do not know how to compute the variance for {self.variance}")

        # Generalised harmonic family — ``history["circular_var_n{n}"]``.
        # The user-supplied ``n_modes`` list is unioned with ``legacy_n``
        # so the legacy key always has a corresponding harmonic computed.
        modes = list(self.n_modes)
        if legacy_n not in modes:
            modes.append(legacy_n)
        z_sums = [np.sum(np.exp(1j * n * angle)) for n in modes]

        # Pack all local quantities into one buffer and reduce in a single call.
        # Layout: 7 scalars + 2 floats per harmonic.
        scalar_block = np.array([
            float(local_n),
            float(local_energy),
            float(local_mom[0]),
            float(local_mom[1]),
            float(local_ang_mom[0]),
            float(local_energy_rot),
            0.0,                      # padding for alignment / future use
        ], dtype=np.float64)
        harm_block = np.empty(2 * len(modes), dtype=np.float64)
        for k, zs in enumerate(z_sums):
            harm_block[2 * k]     = float(zs.real)
            harm_block[2 * k + 1] = float(zs.imag)
        local_buf = np.concatenate([scalar_block, harm_block])
        global_buf = np.zeros_like(local_buf)
        self.comm.Allreduce(local_buf, global_buf, op=MPI.SUM)

        global_n          = global_buf[0]
        global_energy     = global_buf[1]
        global_mom        = global_buf[2:4]
        global_ang_mom    = global_buf[4:5]
        global_energy_rot = global_buf[5]

        # Decode the harmonic block back to per-n complex sums.
        global_z_sums = {}
        for k, n in enumerate(modes):
            global_z_sums[n] = global_buf[7 + 2 * k] + 1j * global_buf[7 + 2 * k + 1]

        mean_u   = global_mom    / global_n
        mean_eta = global_ang_mom / global_n
        temp = (2.0 / (self.dim + 1)) * global_energy / global_n

        R = {n: float(np.abs(z / global_n)) for n, z in global_z_sums.items()}
        legacy_R = R[legacy_n]

        self.history["step"].append(step)
        self.history["temperature"].append(temp)
        self.history["energy"].append(global_energy / global_n)
        if self.interaction_energy is not None:
            E_int = self.interaction_energy(angle)
            L = self.info.get("length", 1.0)
            self.history["interaction_energy"].append(E_int)
            self.history["total_energy"].append(global_energy / global_n + 0.5 * L**2 * E_int)
            self.history["total_energy_rot"].append(global_energy_rot / global_n + 0.5 * L**2 * E_int)
        self.history["momentum_1"].append(np.linalg.norm(mean_u[0]))
        self.history["momentum_2"].append(np.linalg.norm(mean_u[1]))
        self.history["ang_momentum"].append(np.linalg.norm(mean_eta))
        self.history["circular_var"].append(1 - legacy_R)
        for n in self.n_modes:
            self.history[f"circular_var_n{n}"].append(1 - R[n])

        self.swarm.restoreField("orientation")
        self.swarm.restoreField("velocity")
        self.swarm.restoreField("angular_velocity")


        if self.rank == 0:
            with open(f'{self.output_path}/history.pickle', 'wb') as fp:
                pickle.dump(self.history, fp)

        return {
            "N": global_n,
            "mean_u": mean_u,
            "temperature": temp,
            "circular_var": 1 - legacy_R,
            "R": R,
        }

    def maxwellian(self, step):
        """Evaluate the Maxwellian distribution on the velocity-omega grid.

        Uses the temperature and mean momenta recorded at ``step`` in
        ``self.history`` to compute the 3-D Maxwellian on the
        (vx, vy, ω) grid and its marginals along each axis.

        Parameters
        ----------
        step : int
            Index into ``self.history`` from which to read the moments.

        Returns
        -------
        tuple ``(M, M_x, M_y, M_omega)`` where ``M`` is the full 3-D
        distribution and the remaining entries are the marginals.
        """
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
        """Advance the simulation for ``nsteps`` time steps.

        Each step uses second-order Strang splitting:
        ``[DKD](dt/2) → collision × extra_collision → [DKD](dt/2)``
        i.e. D(dt/4)·K(dt/2)·D(dt/4) → collisions → D(dt/4)·K(dt/2)·D(dt/4).
        Diagnostics are computed every step; plots are written every
        ``monitor_every`` steps and at the final step.

        Parameters
        ----------
        nsteps : int
            Number of time steps to run.
        monitor_every : int
            Write histogram plots every this many steps (default 10).
        """
        d = self.diagnostics()
        if self.rank == 0:
            print(
                f"[step 0] N={d['N']}, t = 0.0 "
                f"T={d['temperature']:.6e} "
                f"|u|={np.linalg.norm(d['mean_u']):.6e} "
                f"circ_var={np.linalg.norm(d['circular_var']):.6e} "
            )
        self._construct_grid()
        self.plot_histograms(prefix=f"{self.output_path}/dsmc_0")
        for step in range(1, nsteps + 1):
            if self.transport:
                self.transport_step(dt=0.25*self.dt)         # D(dt/4)
                self.vlasov_kick_step(dt=0.5*self.dt)        # K(dt/2)
                self.transport_step(dt=0.25*self.dt)         # D(dt/4)
            for coll_index in range(self.extra_collision):
                if self.collision_type == "nanbu":
                    self.nanbu_collision_step()
                else:
                    raise ValueError(f"Unknown collision type: {self.collision_type}")
            if self.T_bath is not None:
                self.andersen_thermostat_step()
            if self.transport:
                self.transport_step(dt=0.25*self.dt)         # D(dt/4)
                self.vlasov_kick_step(dt=0.5*self.dt)        # K(dt/2)
                self.transport_step(dt=0.25*self.dt)         # D(dt/4)
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
                if self.rank == 0:
                    self.plot_history(prefix=f"{self.output_path}/dsmc")
            gc.collect()


class CFMZNeedleDSMC:
    """
    DSMC solver for the **non-space-homogeneous** CFMZ kinetic equation:

        ∂f/∂t + v·∇_x f + ω ∂_θ f
                + G[f](θ, x) ∂_v f
                + F[f](θ, x) ∂_ω f
                = Q[f, f]

    Each particle carries:
      - position **x** ∈ R^d (d = 1 or 2; stored on the DMSwarm via the cell DM)
      - translational velocity **v** ∈ R²
      - orientation θ ∈ (0, 2π)
      - angular velocity ω ∈ R

    Strang splitting per step:
        D(dt/4) → K(dt/2) → D(dt/4) → collisions → (Andersen)
                                    → D(dt/4) → K(dt/2) → D(dt/4)
    where D advances both spatial position (X += V·dt) and orientation
    (θ += ω·dt, mod 2π); migration is performed at the end of each D.
    K applies the dual mean-field kick V += G·dt and ω += F·dt with
    forces evaluated against the local-cell orientation density.

    Parameters
    ----------
    opts : dict
        Same keys as ``CFMZNeedleDSMCHomo`` plus:

        - ``test``         (str)   ``"sod_rod"`` (1-D shock tube),
                                    ``"uniform_1d"``, or ``"uniform_2d"``.
        - ``spatial_dim``  (int)   1 or 2; derived from ``test`` if absent.
        - ``bins_theta``   (int)   orientation bins for per-cell density
                                    used by the Vlasov kick (default 32).

    info : dict
        Same keys as ``CFMZNeedleDSMCHomo`` plus a spatial domain block:
        ``Lx`` (1-D) or ``xmin/xmax/ymin/ymax`` (2-D), and
        ``bcs`` ∈ {``"reflective"``, ``"periodic"``}.
        For ``sod_rod``: optional ``right_concentration`` (von Mises κ for
        the right-region orientation distribution; default 2.0).

    vlasov_force : callable or None
        Called as ``vlasov_force(angle, position, density)`` each kick
        substep.  ``density`` is shape ``(nlocal, bins_theta)`` — the
        per-particle copy of its owning cell's orientation histogram.
        Must return per-particle torque, shape ``(nlocal, 1)``.

    translational_force : callable or None
        Same calling convention as ``vlasov_force`` but returns shape
        ``(nlocal, dim)`` per-particle force used to update **v**.

    interaction_energy : callable or None
        Same role as in the homogeneous solver.

    comm : MPI.Comm
        MPI communicator; default ``MPI.COMM_WORLD``.
    """

    def __init__(
        self,
        opts: dict,
        info: dict = {},
        vlasov_force=None,
        translational_force=None,
        interaction_energy=None,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

        self.dim = 2  # translational velocity dimension
        self.nlocal = int(opts["nlocal"])
        self.N = self.nlocal * self.size
        self.nu = opts.get("nu", 1.0)
        self.dt = opts.get("dt", 1e-2)
        self.info = info
        if info.get("cross_section", "maxwell") == "maxwell" and 1 / self.nu < self.dt:
            raise RuntimeError("You have too large of a time-step for the collisional frequency you specified")
        self._nu_max = self.nu
        self.bins = opts.get("bins", 31)
        self.bins_theta = opts.get("bins_theta", 32)
        self.test = opts.get("test", "sod_rod")
        self.variance = opts.get("variance", "circle")
        self.n_modes = [int(n) for n in opts.get("n_modes", [2])]
        # Smectic / positional order parameters ψ_S(k) = |⟨e^{i k·x}⟩|.
        # ``smectic_k`` is a list of tuples; tuple length must match the
        # spatial dimension (1 or 2).  ``None`` disables the diagnostic.
        raw_smectic_k = opts.get("smectic_k", None)
        self.smectic_k = (
            [tuple(float(c) for c in k) for k in raw_smectic_k]
            if raw_smectic_k is not None else None
        )
        self.prefix = opts.get("prefix", "")
        self.extra_collision = opts.get("extra_collision", 1)
        self.collision_type = opts.get("collision_type", "nanbu")
        self.vlasov_force = vlasov_force
        self.translational_force = translational_force
        self.interaction_energy = interaction_energy
        self.transport = opts.get("transport", True)
        self.T_bath = opts.get("T_bath", None)
        self.nu_bath = opts.get("nu_bath", 1.0)
        self.init_at_T_bath = opts.get("init_at_T_bath", True)
        self.dump = "hist"

        # Resolve spatial dimension from `test` if not given explicitly.
        if "spatial_dim" in opts:
            self.spatial_dim = int(opts["spatial_dim"])
        elif self.test in ("uniform_2d",):
            self.spatial_dim = 2
        else:
            self.spatial_dim = 1

        # Mandatory spatial-domain defaults so missing keys do not crash
        # ``_create_mesh``.  Lx is the 1-D length; xmin..ymax bound the 2-D box.
        if self.spatial_dim == 1:
            self.info.setdefault("Lx", 1.0)
        else:
            self.info.setdefault("xmin", 0.0)
            self.info.setdefault("xmax", 1.0)
            self.info.setdefault("ymin", 0.0)
            self.info.setdefault("ymax", 1.0)
        self.info.setdefault("bcs", "reflective")

        self.xlim = 10.0
        self.ylim = 10.0

        self.rng = np.random.default_rng(opts.get("seed", 1234) + self.rank)

        self.history = {
            "step": [],
            "temperature": [],
            "energy": [],
            "momentum_1": [],
            "momentum_2": [],
            "ang_momentum": [],
            "circular_var": [],
        }
        for n in self.n_modes:
            self.history[f"circular_var_n{n}"] = []
        if self.smectic_k is not None:
            for idx in range(len(self.smectic_k)):
                self.history[f"smectic_re_{idx}"] = []
                self.history[f"smectic_im_{idx}"] = []
                self.history[f"smectic_abs_{idx}"] = []
        if interaction_energy is not None:
            self.history["interaction_energy"] = []
            self.history["total_energy"] = []
            self.history["total_energy_rot"] = []

        self.output_path = f'{self.prefix}_output_cfmz_inhomo_{self.collision_type}'
        if self.rank == 0:
            os.makedirs(self.output_path, exist_ok=True)
        self.comm.Barrier()

        self.dm = self._create_mesh()
        self.mesh_dim = self.dm.getDimension()
        self.swarm = self._create_swarm()

        from dsmc.plot import init_plot, plot_cfmz_observables, plot_history
        from .transport_inhomo import transport_step, vlasov_kick_step
        from .collision_inhomo import nanbu_collision_step, andersen_thermostat_step
        from .initial_inhomo import initialize_particles
        from .vtk_export import export_cell_fields_vtk

        self.initialize_particles = initialize_particles.__get__(self)
        self.plot_observables = plot_cfmz_observables.__get__(self)
        self.plot_history = plot_history.__get__(self)
        self.transport_step = transport_step.__get__(self)
        self.vlasov_kick_step = vlasov_kick_step.__get__(self)
        self.andersen_thermostat_step = andersen_thermostat_step.__get__(self)
        self.export_cell_fields_vtk = export_cell_fields_vtk.__get__(self)

        # Collision-operator dispatch.
        # ``info["collision_kind"]`` selects the rod collision kernel:
        #   "boltzmann" (default): cell-local Nanbu, uniform-random
        #     contact normal, no density correction.  Standard DSMC.
        #   "enskog":     cross-cell pair selection, position-derived
        #     contact normal, Parsons-Lee g_PL(η) on the acceptance
        #     weight.  See dsmc.cfmz.collision_enskog_inhomo and
        #     CFMZ.md §14.  Recommended for dense rod systems where
        #     positional correlations matter.
        collision_kind = self.info.get("collision_kind", "boltzmann")
        if collision_kind == "boltzmann":
            self.nanbu_collision_step = nanbu_collision_step.__get__(self)
        elif collision_kind == "enskog":
            from .collision_enskog_inhomo import nanbu_collision_step_enskog_inhomo
            self.nanbu_collision_step = nanbu_collision_step_enskog_inhomo.__get__(self)
            # Cell-sizing constraint for Enskog: cell width must be ≳ L
            # so collision partners always live in the cell + neighbours.
            L = self.info.get("length", 1.0)
            if self.spatial_dim == 1:
                cell_w = self.info["Lx"] / self.bins
            else:
                cell_w = min(
                    (self.info["xmax"] - self.info["xmin"]) / self.bins,
                    (self.info["ymax"] - self.info["ymin"]) / self.bins,
                )
            if cell_w < L:
                if self.rank == 0:
                    print(
                        f"[!] Enskog cell width {cell_w:.3g} < rod length {L:.3g}; "
                        "some collision partners may be missed.  See CFMZ.md §14.4."
                    )
            if self.size > 4 and self.rank == 0:
                print(
                    "[!] Enskog kernel does not currently exchange ghost particles "
                    "across MPI ranks; pair statistics on cell boundaries will be "
                    "lossy with > 4 ranks.  Recommend single-rank or few-rank runs."
                )
        else:
            raise ValueError(
                f"info['collision_kind'] = {collision_kind!r}; expected "
                "'boltzmann' or 'enskog'."
            )

        self.initialize_particles()
        init_plot()

    def _create_mesh(self):
        """Create a DMDA over the physical (x[, y]) domain."""
        nx = self.bins
        if self.spatial_dim == 1:
            Lx = self.info["Lx"]
            self.edges_x = np.linspace(0.0, Lx, nx + 1)
            dm = PETSc.DMDA().create([nx + 1, 2], dof=1, stencil_width=1, comm=self.comm)
            dm.setUp()
            dm.setUniformCoordinates(0.0, Lx, 0.0, 1.0)
        else:
            xmin = self.info["xmin"]
            xmax = self.info["xmax"]
            ymin = self.info["ymin"]
            ymax = self.info["ymax"]
            ny = self.bins
            self.edges_x = np.linspace(xmin, xmax, nx + 1)
            self.edges_y = np.linspace(ymin, ymax, ny + 1)
            dm = PETSc.DMDA().create([nx + 1, ny + 1], dof=1, stencil_width=1, comm=self.comm)
            dm.setUp()
            dm.setUniformCoordinates(xmin, xmax, ymin, ymax)
        return dm

    def _create_swarm(self):
        """Create the DMSwarm with positions + orientation/velocity/ω/weight."""
        swarm = PETSc.DMSwarm().create(comm=self.comm)
        swarm.setDimension(self.mesh_dim)
        swarm.setType(PETSc.DMSwarm.Type.PIC)
        swarm.setCellDM(self.dm)

        swarm.initializeFieldRegister()
        swarm.registerField("orientation", 1, dtype=PETSc.RealType)
        swarm.registerField("velocity", self.dim, dtype=PETSc.RealType)
        swarm.registerField("angular_velocity", 1, dtype=PETSc.RealType)
        swarm.registerField("weight", 1, dtype=PETSc.RealType)
        swarm.finalizeFieldRegister()

        # Generous buffer so transport-step migrations can absorb arrivals.
        swarm.setLocalSizes(self.nlocal, self.N)
        return swarm

    def _construct_grid(self):
        """Build histogram grid edges for velocity (vx, vy) and angular (θ, ω) spaces."""
        self.grid_x = np.linspace(-self.xlim, self.xlim, self.bins + 1)
        self.grid_y = np.linspace(-self.ylim, self.ylim, self.bins + 1)
        self.delta_x = (2 * self.xlim) / (self.bins + 1)
        self.delta_y = (2 * self.ylim) / (self.bins + 1)

        self.grid_angular = np.linspace(self.angular_min, self.angular_max, self.bins + 1)
        self.grid_omega = np.linspace(self.omega_min, self.omega_max, self.bins + 1)
        self.delta_angular = (self.angular_max - self.angular_min) / (self.bins + 1)
        self.delta_omega = (self.omega_max - self.omega_min) / (self.bins + 1)

    def diagnostics(self, step=0):
        """Compute and record global moments (called by all ranks).

        Same global reduction as ``CFMZNeedleDSMCHomo``, but ``self.nlocal``
        is queried from the swarm in case a migration has changed it.

        Tracks an arbitrary harmonic family ``R_n`` per ``opts["n_modes"]``
        (legacy ``circular_var`` is preserved) and an optional smectic /
        positional order parameter family ``ψ_S(k) = |⟨e^{i k·x}⟩|`` per
        ``opts["smectic_k"]`` — useful for detecting columnar / layered
        phases of the discotic solver.
        """
        # Choose the harmonic that drives the legacy ``circular_var``.
        if self.variance == "circle":
            legacy_n = 1
        elif self.variance == "real_projective_plane":
            legacy_n = 2
        else:
            raise RuntimeError(f"[!] Do not know how to compute the variance for {self.variance}")

        modes = list(self.n_modes)
        if legacy_n not in modes:
            modes.append(legacy_n)
        n_smectic = len(self.smectic_k) if self.smectic_k is not None else 0

        # Layout: 7 scalars + 2·|modes| harmonic floats + 2·n_smectic smectic floats.
        n_scalar = 7
        buf_size = n_scalar + 2 * len(modes) + 2 * n_smectic

        self.nlocal = self.swarm.getLocalSize()
        if self.nlocal == 0:
            # Edge case: rank momentarily owns no particles.  Participate in
            # the allreduce with zeros so collective ops stay synchronised.
            local_buf = np.zeros(buf_size, dtype=np.float64)
        else:
            angle = self.swarm.getField("orientation")
            vel = self.swarm.getField("velocity")
            omega = self.swarm.getField("angular_velocity")

            if np.any(angle >= 2 * np.pi) or np.any(angle <= 0):
                raise RuntimeError("[!] Not sticking to the manifold!")

            local_n = self.nlocal
            local_mom = vel.sum(axis=0)
            local_ang_mom = omega.sum(axis=0)
            local_energy_rot = 0.5 * self.info["inertia"] * np.sum(omega * omega)
            local_energy = (0.5 * self.info["mass"] * np.sum(vel * vel) +
                            local_energy_rot)

            z_sums = [np.sum(np.exp(1j * n * angle)) for n in modes]

            scalar_block = np.array([
                float(local_n),
                float(local_energy),
                float(local_mom[0]),
                float(local_mom[1]),
                float(local_ang_mom[0]),
                float(local_energy_rot),
                0.0,
            ], dtype=np.float64)
            harm_block = np.empty(2 * len(modes), dtype=np.float64)
            for k, zs in enumerate(z_sums):
                harm_block[2 * k]     = float(zs.real)
                harm_block[2 * k + 1] = float(zs.imag)

            if n_smectic > 0:
                # Read positions via the swarm's cell-DM coordinate field.
                celldm = self.swarm.getCellDMActive()
                coord_names = celldm.getCoordinateFields()
                pos = self.swarm.getField(coord_names[0])
                X = np.asarray(pos).reshape(local_n, self.mesh_dim)
                smectic_block = np.empty(2 * n_smectic, dtype=np.float64)
                for idx, k_vec in enumerate(self.smectic_k):
                    # Each k_vec has length spatial_dim; X has columns for the
                    # full mesh (spatial + filler).  Take only the first
                    # spatial_dim columns.
                    k_arr = np.asarray(k_vec, dtype=np.float64)
                    phase = X[:, : self.spatial_dim] @ k_arr
                    s = np.exp(1j * phase).sum()
                    smectic_block[2 * idx]     = float(s.real)
                    smectic_block[2 * idx + 1] = float(s.imag)
                self.swarm.restoreField(coord_names[0])
            else:
                smectic_block = np.empty(0, dtype=np.float64)

            local_buf = np.concatenate([scalar_block, harm_block, smectic_block])

            self.swarm.restoreField("orientation")
            self.swarm.restoreField("velocity")
            self.swarm.restoreField("angular_velocity")

        global_buf = np.zeros(buf_size, dtype=np.float64)
        self.comm.Allreduce(local_buf, global_buf, op=MPI.SUM)

        global_n = global_buf[0]
        global_energy = global_buf[1]
        global_mom = global_buf[2:4]
        global_ang_mom = global_buf[4:5]
        global_energy_rot = global_buf[5]

        global_z_sums = {}
        for k, n in enumerate(modes):
            global_z_sums[n] = global_buf[n_scalar + 2 * k] + 1j * global_buf[n_scalar + 2 * k + 1]

        smectic_offset = n_scalar + 2 * len(modes)
        global_smectic = []
        for idx in range(n_smectic):
            re = float(global_buf[smectic_offset + 2 * idx])
            im = float(global_buf[smectic_offset + 2 * idx + 1])
            global_smectic.append(complex(re, im))

        if global_n <= 0:
            raise RuntimeError("[!] Lost all particles during migration.")

        mean_u = global_mom / global_n
        mean_eta = global_ang_mom / global_n
        temp = (2.0 / (self.dim + 1)) * global_energy / global_n

        R = {n: float(np.abs(z / global_n)) for n, z in global_z_sums.items()}
        legacy_R = R[legacy_n]

        self.history["step"].append(step)
        self.history["temperature"].append(temp)
        self.history["energy"].append(global_energy / global_n)
        if self.interaction_energy is not None and self.nlocal > 0:
            angle = self.swarm.getField("orientation")
            E_int = self.interaction_energy(angle)
            self.swarm.restoreField("orientation")
            L = self.info.get("length", 1.0)
            self.history["interaction_energy"].append(E_int)
            self.history["total_energy"].append(global_energy / global_n + 0.5 * L**2 * E_int)
            self.history["total_energy_rot"].append(global_energy_rot / global_n + 0.5 * L**2 * E_int)
        self.history["momentum_1"].append(np.linalg.norm(mean_u[0]))
        self.history["momentum_2"].append(np.linalg.norm(mean_u[1]))
        self.history["ang_momentum"].append(np.linalg.norm(mean_eta))
        self.history["circular_var"].append(1 - legacy_R)
        for n in self.n_modes:
            self.history[f"circular_var_n{n}"].append(1 - R[n])
        for idx, s in enumerate(global_smectic):
            re_n = s.real / global_n
            im_n = s.imag / global_n
            self.history[f"smectic_re_{idx}"].append(re_n)
            self.history[f"smectic_im_{idx}"].append(im_n)
            self.history[f"smectic_abs_{idx}"].append(float(np.hypot(re_n, im_n)))

        if self.rank == 0:
            with open(f'{self.output_path}/history.pickle', 'wb') as fp:
                pickle.dump(self.history, fp)

        return {
            "N": global_n,
            "mean_u": mean_u,
            "temperature": temp,
            "circular_var": 1 - legacy_R,
            "R": R,
            "smectic": [
                {"re": s.real / global_n, "im": s.imag / global_n,
                 "abs": float(np.hypot(s.real, s.imag) / global_n)}
                for s in global_smectic
            ],
        }

    def run(self, nsteps: int, monitor_every: int = 10):
        """Advance the simulation for ``nsteps`` time steps with Strang splitting."""
        d = self.diagnostics()
        if self.rank == 0:
            print(
                f"[step 0] N={d['N']}, t = 0.0 "
                f"T={d['temperature']:.6e} "
                f"|u|={np.linalg.norm(d['mean_u']):.6e} "
                f"circ_var={np.linalg.norm(d['circular_var']):.6e} "
            )
        self._construct_grid()
        self.plot_observables(prefix=f"{self.output_path}/dsmc_0")

        for step in range(1, nsteps + 1):
            if self.transport:
                self.transport_step(dt=0.25 * self.dt)
                self.vlasov_kick_step(dt=0.5 * self.dt)
                self.transport_step(dt=0.25 * self.dt)
            for _ in range(self.extra_collision):
                if self.collision_type == "nanbu":
                    self.nanbu_collision_step()
                else:
                    raise ValueError(f"Unknown collision type: {self.collision_type}")
            if self.T_bath is not None:
                self.andersen_thermostat_step()
            if self.transport:
                self.transport_step(dt=0.25 * self.dt)
                self.vlasov_kick_step(dt=0.5 * self.dt)
                self.transport_step(dt=0.25 * self.dt)
            d = self.diagnostics(step=step)
            if self.rank == 0:
                print(
                    f"[step {step}] N={d['N']}, t ={step*self.dt:.6f} "
                    f"T={d['temperature']:.6e} "
                    f"|u|={np.linalg.norm(d['mean_u']):.6e} "
                    f"circ_var={np.linalg.norm(d['circular_var']):.6e} "
                )
            if step % monitor_every == 0 or step == nsteps:
                self.plot_observables(prefix=f"{self.output_path}/dsmc_{step}")
                if self.rank == 0:
                    self.plot_history(prefix=f"{self.output_path}/dsmc")
            gc.collect()


from .disc import CFMZDiscDSMCHomo, CFMZDiscDSMC  # noqa: E402  (re-export at module scope)
