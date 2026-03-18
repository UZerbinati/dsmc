import numpy as np
def _stick_to_manifold(angle):
    return np.mod(angle, 2*np.pi)

def transport_step(self, dt):
    angle = self.swarm.getField("orientation")
    omega = self.swarm.getField("angular_velocity")

    if self.vlasov_force:
        #TODO: This will have to become a collective action in parallel
        nu_x = np.sum(np.cos(angle))/self.nlocal
        nu_y = np.sum(np.sin(angle))/self.nlocal
        angle_av = (np.atan2(nu_y,nu_x)+2*np.pi)%(2*np.pi)
        omega[:, 0] += self.vlasov_force(angle, angle_av)[:,0]*dt
    angle[:, 0] += omega[:, 0] * dt
    angle[:,0] = _stick_to_manifold(angle[:, 0])

    self.swarm.restoreField("orientation")
    self.swarm.restoreField("angular_velocity")
