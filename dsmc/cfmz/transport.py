import numpy as np

def transport_step(self, dt):
    """Advance orientations by free rotation for time ``dt`` (drift only).

    Pure drift sub-step: θ += ω·dt, then wraps the angle back onto (0, 2π).
    The Vlasov kick (ω update) is handled separately by ``vlasov_kick_step``
    so that the run loop can compose them as D(dt/2)·K(dt)·D(dt/2)
    (Störmer-Verlet), which is second-order accurate instead of the
    first-order KD(dt/2)·KD(dt/2) that results from mixing kick and drift
    inside a single half-step.

    Parameters
    ----------
    dt : float
        Time interval for the drift substep.
    """
    angle = self.swarm.getField("orientation")
    omega = self.swarm.getField("angular_velocity")

    angle[:, 0] += omega[:, 0] * dt
    np.mod(angle[:, 0], 2*np.pi, out=angle[:, 0])

    self.swarm.restoreField("orientation")
    self.swarm.restoreField("angular_velocity")


def vlasov_kick_step(self, dt):
    """Apply the Vlasov mean-field torque for time ``dt`` (kick only).

    ω += F(θ)·dt, where F is the registered ``vlasov_force`` callable
    evaluated at the current (mid-point) orientations.  No-op if no
    Vlasov force is registered.

    Parameters
    ----------
    dt : float
        Time interval for the kick substep.
    """
    if not self.vlasov_force:
        return
    angle = self.swarm.getField("orientation")
    omega = self.swarm.getField("angular_velocity")

    omega[:, 0] += self.vlasov_force(angle)[:, 0] * dt

    self.swarm.restoreField("orientation")
    self.swarm.restoreField("angular_velocity")
