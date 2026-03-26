import numpy as np
import re

def findWholeWord(w):
    """Return a search function that matches whole-word occurrences of ``w`` (case-insensitive)."""
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

def _sample_perturbed_positions_1d(self, N):
    """Sample N angles from the density f(θ) = 1 + A cos(k θ + shift) on [0, 2π].

    Uses a histogram-based inverse-CDF method: the domain is discretised
    into fine cells, each cell is assigned a particle count proportional to
    f evaluated at its midpoint, and samples are drawn uniformly within each
    cell.  The result is shuffled to remove any ordering artefact.
    """
    amp = self.info["initial_angle_amplitude"]
    shift = self.info["initial_angle_shift"]
    k = self.info["initial_angle_wavelength"]

    xmesh = np.linspace(0.0, 2*np.pi, 2001)

    # cell midpoints
    midpoints = 0.5 * (xmesh[1:] + xmesh[:-1])
    deltas = midpoints[1] - midpoints[0]

    # density f(x) = 1 + A cos(k x)
    fx = 1.0 + amp * np.cos(k * midpoints + shift)

    # same normalization as in MATLAB
    cell_area = fx * deltas / (2.0 * np.pi / k)

    counts = cell_area * N
    N_cell = np.floor(counts).astype(np.int64)
    remainder = N - N_cell.sum()
    if remainder > 0:
        fracs = counts - N_cell
        top_up = self.rng.choice(len(N_cell), size=int(remainder), replace=False,
                                 p=fracs / fracs.sum())
        N_cell[top_up] += 1

    sample = np.empty(N)

    start = 0
    for i, count in enumerate(N_cell):
        if count == 0:
            continue
        sample[start:start + count] = xmesh[i] + (xmesh[i + 1] - xmesh[i]) * self.rng.random(count)
        start += count

    # shuffle, like randperm
    self.rng.shuffle(sample)
    return sample

def initialize_particles(self):
    """Initialise particle orientations, velocities, and angular velocities.

    Dispatch based on ``self.test``:
      - ``uniform_angle``:           θ ~ Uniform(0, 2π)
      - ``perturbed_uniform_angle``: θ ~ 1 + A cos(k θ)  (via rejection-free sampling)

    Translational velocities are drawn from Uniform(-2, 2)² and angular
    velocities from Uniform(-1, 1) for all test cases.
    """
    angle = self.swarm.getField("orientation")
    vel = self.swarm.getField("velocity")
    angular_vel = self.swarm.getField("angular_velocity")
    wgt = self.swarm.getField("weight")
    if findWholeWord("uniform_angle")(self.test):
        angle[:] = self.rng.uniform(size=(self.nlocal, 1), low=0, high=2*np.pi)
    elif findWholeWord("perturbed_uniform_angle")(self.test):
        angle[:,0] = _sample_perturbed_positions_1d(self, self.nlocal)
    else:
        raise RuntimeError("[!] How shell I initialise the angular data ?")
    vel[:] = self.rng.uniform(size=(self.nlocal, 2), low=-0.5, high=0.5)
    angular_vel[:] = self.rng.uniform(size=(self.nlocal, 1), low=-0.25, high=0.25)
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
        raise RuntimeError("[!] Larger angular min than angular max.")
    if self.omega_min > self.omega_max:
        raise RuntimeError("[!] Larger omega min than omega max.")
