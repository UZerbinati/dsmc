import numpy as np


def _reflect_1d(x, v, xmin, xmax):
    """Elastic reflection at domain boundaries [xmin, xmax]."""
    left = x < xmin
    right = x > xmax
    x[left] = 2.0 * xmin - x[left]
    v[left] = -v[left]
    x[right] = 2.0 * xmax - x[right]
    v[right] = -v[right]


def transport_step(self, dt):
    self.swarm.sortGetAccess()
    celldm = self.swarm.getCellDMActive()
    coord_names = celldm.getCoordinateFields()
    pos = self.swarm.getField(coord_names[0])
    vel = self.swarm.getField("velocity")

    X = pos.reshape(self.nlocal, self.mesh_dim)
    V = vel.reshape(self.nlocal, self.dim)

    for d in range(self.effective_dim):
        X[:, d] += V[:, d] * dt
    _reflect_1d(X[:, 0], V[:, 0], 0.0, self.info["Lx"])

    self.swarm.restoreField(coord_names[0])
    self.swarm.restoreField("velocity")
    self.swarm.sortRestoreAccess()
    self.swarm.migrate(remove_sent_points=True)
    self.nlocal = self.swarm.getLocalSize()
