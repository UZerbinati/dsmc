import numpy as np
def initialize_particles(self):
    angle = self.swarm.getField("orientation")
    vel = self.swarm.getField("velocity")
    angular_vel = self.swarm.getField("angular_velocity")
    wgt = self.swarm.getField("weight")
    if "uniform_angle" in self.test:
        angle[:] = self.rng.uniform(size=(self.nlocal, 1), low=0, high=2*np.pi)
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
