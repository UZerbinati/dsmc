import numpy as np
def _stick_to_manifold(angle):
    return np.mod(angle, 2*np.pi)

def transport_step(self, dt):
    angle = self.swarm.getField("orientation")
    omega = self.swarm.getField("angular_velocity")

    if self.vlasov_force:
        omega[:, 0] += self.vlasov_force(angle)[:,0]*dt
    angle[:, 0] += omega[:, 0] * dt
    angle[:,0] = _stick_to_manifold(angle[:, 0])

    self.swarm.restoreField("orientation")
    self.swarm.restoreField("angular_velocity")
