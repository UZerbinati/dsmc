"""
Plot the isotropic–nematic phase diagram from the results of test_27.

Reads the history.pickle produced by each temperature in the sweep and
plots the steady-state circular variance (real projective plane)

    σ²  =  1 − |⟨exp(2iθ)⟩|

against T_bath.  σ² ≈ 1 indicates the isotropic phase; σ² ≈ 0 the
nematic phase.

Usage
-----
    python plot_phase_diagram_27.py [output_root] [collision_type]

Defaults: output_root = "output/test_27",  collision_type = "nanbu".
"""

import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ #
# Configuration                                                        #
# ------------------------------------------------------------------ #
output_root    = sys.argv[1] if len(sys.argv) > 1 else "output/test_27"
collision_type = sys.argv[2] if len(sys.argv) > 2 else "nanbu"

T_bath_values = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]
T_c = 8.0 / np.pi   # spinodal estimate for L = sqrt(12)

# ------------------------------------------------------------------ #
# Load histories                                                       #
# ------------------------------------------------------------------ #
T_found   = []
circ_vars = []

for T_b in T_bath_values:
    path = os.path.join(
        output_root,
        f"T_{T_b:.2f}_output_cfmz_{collision_type}",
        "history.pickle",
    )
    if not os.path.exists(path):
        print(f"[skip] {path} not found")
        continue
    with open(path, "rb") as f:
        h = pickle.load(f)
    circ_vars.append(float(h["circular_var"][-1]))
    T_found.append(T_b)
    print(f"T_bath={T_b:.2f}  circular_var={circ_vars[-1]:.4f}")

if not T_found:
    sys.exit("No history files found — has test_27 been run?")

T_found   = np.array(T_found)
circ_vars = np.array(circ_vars)

# ------------------------------------------------------------------ #
# Phase diagram plot                                                   #
# ------------------------------------------------------------------ #
fig, ax = plt.subplots(figsize=(5, 3.5))

ax.plot(T_found, circ_vars, "o-", color="black", linewidth=1.5, markersize=5,
        label=r"$\sigma^2$ (DSMC)")
ax.axvline(T_c, color="gray", linestyle="--", linewidth=1.0,
           label=rf"$T_c = 8/\pi \approx {T_c:.2f}$ (spinodal)")

ax.set_xlabel(r"$T_{\mathrm{bath}}$")
ax.set_ylabel(r"circular variance $\sigma^2 = 1 - |\langle e^{2i\theta}\rangle|$")
ax.set_ylim(-0.05, 1.05)
ax.tick_params(which="both", direction="in", top=True, right=True)
ax.legend(frameon=False, fontsize=8)

# Phase labels
isotropic_x = (T_c + T_found.max()) / 2
nematic_x   = (T_found.min() + T_c) / 2 - 0.4
ax.text(isotropic_x, 0.92, "isotropic", ha="center", va="top",   fontsize=9, color="dimgray")
ax.text(nematic_x,   0.15, "nematic",   ha="center", va="bottom", fontsize=9, color="dimgray")

fig.tight_layout()

out_pdf = os.path.join(output_root, "phase_diagram.pdf")
out_png = os.path.join(output_root, "phase_diagram.png")
fig.savefig(out_pdf)
fig.savefig(out_png, dpi=400)
print(f"\nSaved {out_pdf}")
print(f"Saved {out_png}")
plt.show()
