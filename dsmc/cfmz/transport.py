import numpy as np
def _stick_to_manifold(angle):
    """Wrap angle back onto [0, 2π)."""
    return np.mod(angle, 2*np.pi)

def transport_step(self, dt):
    """Advance orientations by free rotation for time ``dt``.

    If a Vlasov force is registered it is applied to ω first (explicit
    Euler), then θ is updated as θ += ω·dt, and finally the angle is
    wrapped back onto (0, 2π) by ``_stick_to_manifold``.

    Parameters
    ----------
    dt : float
        Time interval for the rotation substep.
    """
    angle = self.swarm.getField("orientation")
    omega = self.swarm.getField("angular_velocity")

    if self.vlasov_force:
        omega[:, 0] += self.vlasov_force(angle)[:,0]*dt
    angle[:, 0] += omega[:, 0] * dt
    angle[:,0] = _stick_to_manifold(angle[:, 0])

    self.swarm.restoreField("orientation")
    self.swarm.restoreField("angular_velocity")
