import numpy as np

def transport_step(self, dt):
    """Advance orientations by free rotation for time ``dt``.

    If a Vlasov force is registered it is applied to ω first (explicit
    Euler), then θ is updated as θ += ω·dt, and finally the angle is
    wrapped back onto (0, 2π) in-place.

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
    np.mod(angle[:, 0], 2*np.pi, out=angle[:, 0])

    self.swarm.restoreField("orientation")
    self.swarm.restoreField("angular_velocity")
