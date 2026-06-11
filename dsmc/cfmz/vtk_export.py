"""
Per-cell VTK export for the CFMZ inhomogeneous solver.

Writes ParaView-readable ``vtkRectilinearGrid`` (``.vtr``) files
containing the spatial grid of CFMZNeedleDSMC / CFMZDiscDSMC and a set
of per-cell aggregated fields:

  density           scalar   particles per cell / cell area (or length, in 1-D)
  mean_velocity     3-vector ⟨v⟩_cell, padded with z=0
  mean_orientation  3-vector director from the Q-tensor — (cos θ̄, sin θ̄, 0)
  local_R2          scalar   R_2 = √(⟨cos 2θ⟩² + ⟨sin 2θ⟩²) on the cell
  local_temperature scalar   T_cell = m(⟨v²⟩ − |⟨v⟩|²) (translational only)
  cell_eta          scalar   Parsons-Lee argument η = (π/4) ρ L² on the cell

Output is **ASCII** XML (no binary base64 / streamed appended data) so
the format works with the Python stdlib alone — no PyVista, no `vtk`
package dependency.  For the simulation sizes typical of CFMZ
(~10⁴–10⁵ cells, 6 fields), the file size is small and ParaView reads
it fast.

Visualisation grid (`vis_bins`)
-------------------------------
By default the output uses ``self.bins`` — the Enskog kernel mesh.
That mesh is constrained by ``dx ≥ L`` (rod length) so neighbour-cell
pair sampling captures all contacts, but a smectic layer also has
period λ = L, so density-stripe visualisation aliases on the kernel
grid (Nyquist).  Pass ``vis_bins`` to override the VTK mesh
independently of ``self.bins``; the simulation state is untouched.
A reasonable choice is ``vis_bins = 8 · n_layers`` (so
``dx_vis = L / 8`` — well below Nyquist).

ParaView pipeline
-----------------
1. ``paraview output/<prefix>.pvd`` (the time-series collection
   produced by :func:`write_pvd_collection`).
2. Cell-colour the underlying grid by ``density`` to see smectic
   stripes, or by ``local_R2`` to confirm uniform nematic alignment.
3. Apply a *Glyph* filter on ``mean_orientation`` (centred glyphs,
   scale by ``local_R2``).
4. For an animation, use the time slider; ``TimeValue`` is stored as a
   ``FieldData`` entry on each file.

Method binding
--------------
``export_cell_fields_vtk(self, prefix, smectic_k=None, time=None,
vis_bins=None)`` is bound on ``CFMZNeedleDSMC`` (and inherited by
``CFMZDiscDSMC``) at construction time, mirroring the other method
bindings in ``dsmc.cfmz.CFMZNeedleDSMC.__init__``.

The function is a *no-op on non-rank-0 ranks* after the MPI-Reduce, so
all ranks must call it for the reductions to be collectively
synchronised — same protocol as ``diagnostics()``.

Note on `smectic_k`
-------------------
Per-cell smectic projections (``⟨cos 2θ · e^{i k·x}⟩_cell``) were
emitted in earlier versions but turned out to be aliased by the same
cell-width-vs-layer-width constraint as ``density`` on the Enskog
mesh.  The kwarg is kept in the signature for backward compatibility
but is ignored.  The authoritative diagnostic is the global
``smectic_abs_<idx>`` history series (computed in ``diagnostics()``
without per-cell averaging).
"""
import os
import numpy as np
from mpi4py import MPI


def _xml_array(name, dtype, n_components, data):
    """Return an XML <DataArray> element for a numpy array.

    Parameters
    ----------
    name : str
        VTK array name.
    dtype : str
        VTK type string (``"Float64"`` is what we use uniformly).
    n_components : int
        Number of components per tuple.  Scalars use 1; 3-vectors use 3.
    data : np.ndarray
        Flat array, length n_cells * n_components, in cell-major
        ordering matching the structured grid extent.

    Returns
    -------
    str — XML lines for the array, including opening / closing tags.
    """
    flat = np.asarray(data, dtype=np.float64).ravel()
    payload = " ".join(f"{v:.10e}" for v in flat)
    if n_components == 1:
        header = f'<DataArray type="{dtype}" Name="{name}" format="ascii">'
    else:
        header = (
            f'<DataArray type="{dtype}" Name="{name}" '
            f'NumberOfComponents="{n_components}" format="ascii">'
        )
    return f"{header}{payload}</DataArray>\n"


def _write_vtr(path, x_edges, y_edges, fields, time=None):
    """Write a 2-D ``vtkRectilinearGrid`` XML file.

    Parameters
    ----------
    path : str
        Output filename, including ``.vtr`` extension.
    x_edges, y_edges : np.ndarray
        Cell-edge coordinates of length nx+1 and ny+1, monotonically
        increasing.  For a 1-D simulation pass ``y_edges = [0, 1]`` so
        a degenerate one-cell-thick slab is produced.
    fields : list[(name, n_components, data)]
        Per-cell field tuples; `data` has length ``nx * ny *
        n_components`` in cell-major ordering (the standard VTK
        rectilinear-grid CellData layout: x varies fastest).
    time : float or None
        Simulation time stamp, written as a ``FieldData`` entry.
    """
    nx = x_edges.size - 1
    ny = y_edges.size - 1
    extent = f"0 {nx} 0 {ny} 0 0"
    z_edges = np.array([0.0], dtype=np.float64)

    with open(path, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write(
            '<VTKFile type="RectilinearGrid" version="0.1" '
            'byte_order="LittleEndian">\n'
        )
        f.write(f'<RectilinearGrid WholeExtent="{extent}">\n')
        if time is not None:
            f.write('<FieldData>\n')
            f.write(_xml_array("TimeValue", "Float64", 1, np.array([time])))
            f.write('</FieldData>\n')
        f.write(f'<Piece Extent="{extent}">\n')
        f.write('<CellData>\n')
        for name, n_components, data in fields:
            f.write(_xml_array(name, "Float64", n_components, data))
        f.write('</CellData>\n')
        f.write('<Coordinates>\n')
        f.write(_xml_array("X", "Float64", 1, x_edges))
        f.write(_xml_array("Y", "Float64", 1, y_edges))
        f.write(_xml_array("Z", "Float64", 1, z_edges))
        f.write('</Coordinates>\n')
        f.write('</Piece>\n')
        f.write('</RectilinearGrid>\n')
        f.write('</VTKFile>\n')


def export_cell_fields_vtk(self, prefix, smectic_k=None, time=None,
                           vis_bins=None):
    """Write per-cell fields of the inhomogeneous solver to a VTK file.

    Collective: must be called on all MPI ranks.  Only rank 0 writes
    the file.

    Parameters
    ----------
    prefix : str
        Output filename without the ``.vtr`` extension.  The actual
        path will be ``f"{prefix}.vtr"``.
    smectic_k : iterable, optional
        Accepted for backward compatibility but ignored.  Per-cell
        smectic projections were dropped because they alias on the
        Enskog grid; use the global ``smectic_abs_<idx>`` history
        series instead.
    time : float or None
        Simulation time stored as ``FieldData/TimeValue`` so ParaView
        animates files in a directory in correct temporal order.
    vis_bins : int or None
        Override the spatial bin count for the **visualisation grid
        only** — independent of the Enskog kernel's ``self.bins``.
        Useful when the Enskog cell-sizing constraint forces
        ``dx ≥ L`` while smectic-stripe visualisation needs
        ``dx ≪ L/2`` (Nyquist).  Pass e.g. ``vis_bins = 8 * n_layers``
        to resolve the layers cleanly.  Default ``None`` keeps
        ``vis_bins = self.bins``.
    """
    del smectic_k  # accepted but unused (see docstring)

    comm = self.comm
    rank = comm.Get_rank()
    nlocal = self.swarm.getLocalSize()

    # ------------------------------------------------------------------
    # Resolve grid layout.  ``vis_bins`` overrides only the VTK output
    # mesh — the Enskog kernel and the simulation state are untouched.
    # For 1-D simulations we emit a single-row rectilinear grid (ny=1)
    # so ParaView still gets a 2-D-shaped file.
    # ------------------------------------------------------------------
    n_bins = int(vis_bins) if vis_bins else int(self.bins)
    if self.spatial_dim == 1:
        Lx = self.info["Lx"]
        nx = n_bins
        dx = Lx / nx
        x_edges = np.linspace(0.0, Lx, nx + 1)
        ny = 1
        y_edges = np.array([0.0, 1.0])
        xmin = 0.0
        ymin = 0.0
        cell_volume = dx               # length, in 1-D
    else:
        xmin = self.info["xmin"]
        xmax = self.info["xmax"]
        ymin = self.info["ymin"]
        ymax = self.info["ymax"]
        nx = n_bins
        ny = n_bins
        dx = (xmax - xmin) / nx
        dy = (ymax - ymin) / ny
        x_edges = np.linspace(xmin, xmax, nx + 1)
        y_edges = np.linspace(ymin, ymax, ny + 1)
        cell_volume = dx * dy

    n_cells = nx * ny

    # ------------------------------------------------------------------
    # Per-rank aggregation.  Empty-rank fast path is just zeros.
    # ------------------------------------------------------------------
    counts   = np.zeros(n_cells, dtype=np.float64)
    sum_vx   = np.zeros(n_cells, dtype=np.float64)
    sum_vy   = np.zeros(n_cells, dtype=np.float64)
    sum_v2   = np.zeros(n_cells, dtype=np.float64)
    sum_cos2 = np.zeros(n_cells, dtype=np.float64)
    sum_sin2 = np.zeros(n_cells, dtype=np.float64)

    if nlocal > 0:
        # Read positions (via cell-DM coord field), velocity, orientation.
        celldm = self.swarm.getCellDMActive()
        coord_names = celldm.getCoordinateFields()
        pos = self.swarm.getField(coord_names[0])
        vel = self.swarm.getField("velocity")
        angle = self.swarm.getField("orientation")
        try:
            X = np.asarray(pos).reshape(nlocal, self.mesh_dim)
            V = np.asarray(vel).reshape(nlocal, self.dim)
            theta = np.asarray(angle).ravel()

            # Map each particle to a global cell index.  Clip into
            # [0, n) to swallow tiny round-off at the right boundary.
            x_local = X[:, 0]
            i_idx = np.clip(((x_local - xmin) / dx).astype(np.int64), 0, nx - 1)
            if self.spatial_dim == 1:
                cell_id = i_idx
            else:
                y_local = X[:, 1]
                j_idx = np.clip(((y_local - ymin) / dy).astype(np.int64), 0, ny - 1)
                cell_id = i_idx + j_idx * nx

            # Aggregate.  np.bincount is the parallelism-safe choice
            # here since cell_id may have repeats.
            counts = np.bincount(cell_id, minlength=n_cells).astype(np.float64)
            sum_vx = np.bincount(cell_id, weights=V[:, 0], minlength=n_cells)
            sum_vy = np.bincount(cell_id, weights=V[:, 1], minlength=n_cells)
            # Translational kinetic-energy moment for the per-cell
            # temperature T_cell = m(⟨v²⟩ − |⟨v⟩|²); summed over the
            # available translational components (2 in 2-D, 1 in 1-D).
            v2 = np.einsum("ij,ij->i",
                           V[:, : self.spatial_dim],
                           V[:, : self.spatial_dim])
            sum_v2 = np.bincount(cell_id, weights=v2, minlength=n_cells)
            sum_cos2 = np.bincount(cell_id, weights=np.cos(2.0 * theta),
                                   minlength=n_cells)
            sum_sin2 = np.bincount(cell_id, weights=np.sin(2.0 * theta),
                                   minlength=n_cells)
        finally:
            self.swarm.restoreField(coord_names[0])
            self.swarm.restoreField("velocity")
            self.swarm.restoreField("orientation")

    # ------------------------------------------------------------------
    # MPI-Reduce all aggregates to rank 0.
    # ------------------------------------------------------------------
    def _reduce(arr):
        out = np.zeros_like(arr)
        comm.Reduce(arr, out, op=MPI.SUM, root=0)
        return out

    counts_g   = _reduce(counts)
    sum_vx_g   = _reduce(sum_vx)
    sum_vy_g   = _reduce(sum_vy)
    sum_v2_g   = _reduce(sum_v2)
    sum_cos2_g = _reduce(sum_cos2)
    sum_sin2_g = _reduce(sum_sin2)

    if rank != 0:
        return

    # ------------------------------------------------------------------
    # Per-cell averages on rank 0.  Empty cells get zeros.
    # ------------------------------------------------------------------
    valid = counts_g > 0
    inv_count = np.zeros(n_cells, dtype=np.float64)
    inv_count[valid] = 1.0 / counts_g[valid]

    # Density and cell_eta are *physical* (extensive) quantities — they
    # depend on the particle-weight factor F_N = self.particle_weight.
    # Intensive averages (mean_vx, mean_vy, local_T, local_R2) are
    # F_N-independent.
    weight = float(getattr(self, "particle_weight", 1.0))
    density = weight * counts_g / cell_volume
    mean_vx = sum_vx_g * inv_count
    mean_vy = sum_vy_g * inv_count
    cos2_mean = sum_cos2_g * inv_count
    sin2_mean = sum_sin2_g * inv_count
    local_R2 = np.hypot(cos2_mean, sin2_mean)

    # Per-cell translational temperature (same convention as the legacy
    # plot_cfmz_observables): T = m(⟨v²⟩ − |⟨v⟩|²).  Empty / single-
    # particle cells get T = 0 so VTK colour scales aren't blown out.
    mass = float(self.info.get("mass", 1.0))
    v2_mean = sum_v2_g * inv_count
    local_T = mass * (v2_mean - mean_vx ** 2 - mean_vy ** 2)
    local_T = np.where(counts_g > 1, local_T, 0.0)

    # Q-tensor director: θ̄ = (1/2) arctan2(⟨sin 2θ⟩, ⟨cos 2θ⟩).
    theta_bar = 0.5 * np.arctan2(sin2_mean, cos2_mean)
    mean_ox = np.cos(theta_bar)
    mean_oy = np.sin(theta_bar)
    # Zero out the orientation glyph in empty cells (otherwise an
    # arbitrary unit arrow would appear there).
    mean_ox[~valid] = 0.0
    mean_oy[~valid] = 0.0

    L = self.info.get("length", 1.0)
    cell_eta = (np.pi / 4.0) * density * L * L

    # Pack mean velocity / orientation as 3-vectors (z = 0).
    zero = np.zeros(n_cells, dtype=np.float64)
    mean_velocity = np.column_stack((mean_vx, mean_vy, zero)).ravel()
    mean_orient   = np.column_stack((mean_ox, mean_oy, zero)).ravel()

    fields = [
        ("density",          1, density),
        ("mean_velocity",    3, mean_velocity),
        ("mean_orientation", 3, mean_orient),
        ("local_R2",         1, local_R2),
        ("local_temperature", 1, local_T),
        ("cell_eta",         1, cell_eta),
    ]

    out_dir = os.path.dirname(prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_path = f"{prefix}.vtr"
    _write_vtr(out_path, x_edges, y_edges, fields, time=time)

    # Track this snapshot for an optional PVD collection later (rank 0 only).
    if not hasattr(self, "_vtk_snapshots"):
        self._vtk_snapshots = []
    self._vtk_snapshots.append((float(time) if time is not None else
                                float(len(self._vtk_snapshots)),
                                out_path))


def write_pvd_collection(self, pvd_path, entries=None):
    """Write a ``.pvd`` time-series index referencing the per-step ``.vtr`` files.

    A PVD file is a single XML "collection" that ParaView opens as one
    dataset with a time slider — the underlying ``.vtr`` files are not
    duplicated, just referenced.

    Parameters
    ----------
    pvd_path : str
        Output path.  ``.pvd`` extension is appended if missing.
    entries : iterable of (time, vtr_path) tuples, optional
        If omitted, uses ``self._vtk_snapshots`` accumulated by
        ``export_cell_fields_vtk``.

    No-op on non-rank-0 ranks.
    """
    if self.comm.Get_rank() != 0:
        return
    if entries is None:
        entries = getattr(self, "_vtk_snapshots", [])
    if not entries:
        return
    if not pvd_path.endswith(".pvd"):
        pvd_path = pvd_path + ".pvd"
    out_dir = os.path.dirname(pvd_path)
    base = out_dir if out_dir else "."

    with open(pvd_path, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1" '
                'byte_order="LittleEndian">\n')
        f.write('<Collection>\n')
        for t, vtr in entries:
            # Reference each .vtr by a path relative to the .pvd so the
            # bundle is portable (move the directory, the index still works).
            rel = os.path.relpath(vtr, start=base)
            f.write(f'<DataSet timestep="{t:.10e}" group="" part="0" '
                    f'file="{rel}"/>\n')
        f.write('</Collection>\n')
        f.write('</VTKFile>\n')
