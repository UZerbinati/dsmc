import numpy as np
import re

def findWholeWord(w):
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

def _sample_perturbed_positions_1d(self, N):
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

    N_cell = np.round(cell_area * N).astype(np.int64)

    if N_cell.sum() != N:
        raise ValueError(
            f"Particle count mismatch: sum(N_cell)={N_cell.sum()} but N={N}"
        )

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
        raise RuntimeError("[!] Larger angular min than angular max.")
    if self.omega_min > self.omega_max:
        raise RuntimeError("[!] Larger omega min than omega max.")
