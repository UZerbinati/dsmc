"""
Shared utilities used by all DSMC classes.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def init_plot():
    """Apply SIAM-style matplotlib defaults."""
    mpl.rcParams.update({
        "font.family":       "serif",
        "mathtext.fontset":  "cm",
        "font.size":         12,
        "axes.labelsize":    14,
        "xtick.labelsize":   11,
        "ytick.labelsize":   11,
        "axes.linewidth":    0.8,
        "lines.linewidth":   1.5,
        "xtick.direction":   "in",
        "ytick.direction":   "in",
        "xtick.major.size":  4,
        "ytick.major.size":  4,
        "xtick.minor.size":  2,
        "ytick.minor.size":  2,
    })

# Fixed axes box dimensions (inches) shared by all plots.
_AX_W,   _AX_H   = 4.0,  2.6
_ML, _MR, _MT, _MB = 0.90, 0.30, 0.20, 0.68
_CBAR_W, _CBAR_PAD  = 0.18, 0.10

def fig_axes(colorbar=False):
    """Return (fig, ax[, cax]) with a fixed axes box of _AX_W × _AX_H inches.

    Every plot produced by this helper has an identical axes box regardless
    of whether a colorbar is present; the figure is simply made wider to
    accommodate it.
    """
    r  = _MR + (_CBAR_PAD + _CBAR_W if colorbar else 0)
    fw = _ML + _AX_W + r
    fh = _MT + _AX_H + _MB
    fig = plt.figure(figsize=(fw, fh))
    ax  = fig.add_axes([_ML/fw, _MB/fh, _AX_W/fw, _AX_H/fh])
    ax.yaxis.get_major_formatter().set_useOffset(False)
    ax.xaxis.get_major_formatter().set_useOffset(False)
    if colorbar:
        cax = fig.add_axes([(_ML + _AX_W + _CBAR_PAD)/fw, _MB/fh, _CBAR_W/fw, _AX_H/fh])
        return fig, ax, cax
    return fig, ax, None


def paraview_cool_to_warm_extended():
    """ParaView 'Cool to Warm (Extended)' colormap."""
    pts = [
        (0.000000, (0.000000, 0.000000, 0.349020)),
        (0.031250, (0.039216, 0.062745, 0.380392)),
        (0.062500, (0.062745, 0.117647, 0.411765)),
        (0.093750, (0.090196, 0.184314, 0.450980)),
        (0.125000, (0.125490, 0.262745, 0.501961)),
        (0.156250, (0.160784, 0.337255, 0.541176)),
        (0.187500, (0.200000, 0.396078, 0.568627)),
        (0.218750, (0.239216, 0.454902, 0.600000)),
        (0.250000, (0.286275, 0.521569, 0.650980)),
        (0.281250, (0.337255, 0.592157, 0.701961)),
        (0.312500, (0.388235, 0.654902, 0.749020)),
        (0.343750, (0.466667, 0.737255, 0.819608)),
        (0.375000, (0.572549, 0.819608, 0.878431)),
        (0.406250, (0.654902, 0.866667, 0.909804)),
        (0.437500, (0.752941, 0.917647, 0.941176)),
        (0.468750, (0.823529, 0.956863, 0.968627)),
        (0.500000, (0.941176, 0.984314, 0.988235)),
        (0.500000, (0.988235, 0.960784, 0.901961)),
        (0.520000, (0.988235, 0.945098, 0.850980)),
        (0.540000, (0.980392, 0.898039, 0.784314)),
        (0.562500, (0.968627, 0.835294, 0.698039)),
        (0.593750, (0.949020, 0.733333, 0.588235)),
        (0.625000, (0.929412, 0.650980, 0.509804)),
        (0.656250, (0.909804, 0.564706, 0.435294)),
        (0.687500, (0.878431, 0.458824, 0.352941)),
        (0.718750, (0.839216, 0.388235, 0.286275)),
        (0.750000, (0.760784, 0.294118, 0.211765)),
        (0.781250, (0.701961, 0.211765, 0.168627)),
        (0.812500, (0.650980, 0.156863, 0.129412)),
        (0.843750, (0.600000, 0.094118, 0.094118)),
        (0.875000, (0.549020, 0.066667, 0.098039)),
        (0.906250, (0.501961, 0.050980, 0.125490)),
        (0.937500, (0.450980, 0.054902, 0.172549)),
        (0.968750, (0.400000, 0.054902, 0.192157)),
        (1.000000, (0.349020, 0.070588, 0.211765)),
    ]
    return LinearSegmentedColormap.from_list("pv_cool_to_warm_extended", pts, N=256)


pv_cmap = paraview_cool_to_warm_extended()


# ---------------------------------------------------------------------------
# DMSwarm cell utilities
# ---------------------------------------------------------------------------

def build_cell_lists(cells):
    """
    Group local particle indices by cell ID.

    Parameters
    ----------
    cells : 1-D int32 array of length n_local
        Cell ID for each local particle.

    Returns
    -------
    dict mapping cell_id -> 1-D array of particle indices in that cell.
    """
    if cells.size == 0:
        return {}
    order = np.argsort(cells)
    cells_sorted = cells[order]
    starts = np.flatnonzero(
        np.r_[True, cells_sorted[1:] != cells_sorted[:-1]]
    )
    ends = np.r_[starts[1:], len(cells_sorted)]
    cell_lists = {}
    for a, b in zip(starts, ends):
        c = int(cells_sorted[a])
        cell_lists[c] = order[a:b]
    return cell_lists


def get_particle_cells(self):
    """Return the cell-ID array for all local particles (copied)."""
    celldm = self.swarm.getCellDMActive()
    cellid_name = celldm.getCellID()
    arr = self.swarm.getField(cellid_name)
    try:
        return np.asarray(arr).reshape(-1).astype(np.int32).copy()
    finally:
        self.swarm.restoreField(cellid_name)
