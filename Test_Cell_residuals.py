""" Checks for CubicGalileonForecasts.py: C(ell) residuals of the data vector and of the model.

Plots, for every 3x2pt auto/cross-correlation, the fractional residual with respect to GR

    (C_ell - C_ell^GR) / C_ell^GR

for (a) the data vector built by the forecast script and (b) the model that the likelihood
evaluates at the input-universe cosmological parameters. Both are compared to the same GR
reference, computed at the input-universe cosmology. Grey bands are the 1-sigma errors from
the diagonal of the covariance matrix, and the red shading marks points removed by the scale cuts.

For data type 1 (emulator) data and model are built the same way, so the curves must lie on top
of each other. For type 0 (validation boost) and type 2 (N-body) the gap between them is the
emulator error.

Usage (from this directory):
    python Test_Cell_residuals.py ini_files/config_run_ell1500_k0p2.yaml
    python Test_Cell_residuals.py <config.yaml> --out Figures/my_plot.pdf --xmax 5000 --show
"""
import argparse
import os
import sys

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("config", help="forecast config yaml, same file you would pass to CubicGalileonForecasts.py")
parser.add_argument("--out", default="Figures/CuGal_Cells_residuals_data_vs_model.pdf", help="output figure")
parser.add_argument("--xmax", type=float, default=5000, help="upper ell limit of the panels")
parser.add_argument("--show", action="store_true", help="open the figure window")
args = parser.parse_args()

# CubicGalileonForecasts.py reads sys.argv[1] and uses paths relative to its own directory when imported
HERE = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.relpath(os.path.abspath(args.config), HERE)
out_path = os.path.abspath(args.out)
os.chdir(HERE)
sys.path.insert(0, HERE)
sys.argv = [sys.argv[0], config_path]

import matplotlib
if not args.show:
    matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# runs the whole set-up of the forecast (data vector, covariance, scale cuts); main() is not called
import CubicGalileonForecasts as cgf

ccl = cgf.ccl
PARAM_ORDER = ["Omega_m", "f_phi", "1e9As", "h", "ns", "Omega_b", "b1", "b2", "b3", "b4", "b5"]
N_ELL = cgf.ell_bin_num_mockdata


########################## model / GR vectors ##########################

def model_datavector(theta):
    """ Model 3x2pt vector, ordered [kk, delk, deldel]. Mirrors CubicGalileonFunctions.loglikelihood. """
    Omega_m, f_phi, A_s1e9, h, n_s, Omega_b, b1, b2, b3, b4, b5 = [float(theta[key]) for key in PARAM_ORDER]
    bias = np.array([b * np.ones(len(cgf.z)) for b in (b1, b2, b3, b4, b5)])
    cosmo = ccl.Cosmology(Omega_c=Omega_m - Omega_b, Omega_b=Omega_b, h=h, n_s=n_s, A_s=A_s1e9 * 1e-9)

    a_arr, UE_arr, coupling_arr = cgf.CuGal_initialize(f_phi, cosmo)
    if cgf.is_model_linear:
        Pk_GR = cgf.Get_Pk2D_obj_kk_GR_lin(cosmo)
    else:
        Pk_GR = cgf.Get_Pk2D_obj_kk_GR_nl(cosmo)
    # NB: loglikelihood always uses the non-"_lin" deldel here (its "_lin" call is overwritten)
    Pk_gg = cgf.Pk_2D_obj_CuGal_deldel(Pk_GR, f_phi, cosmo, a_arr, UE_arr, coupling_arr)
    Pk_kg = cgf.Pk_2D_obj_CuGal_delk(Pk_gg, f_phi, cosmo, a_arr, UE_arr, coupling_arr)
    Pk_kk = cgf.Pk_2D_obj_CuGal_kk(Pk_gg, f_phi, cosmo, a_arr, UE_arr, coupling_arr)

    Bs, Bl = cgf.Binned_distribution_source, cgf.Binned_distribution_lens
    ell_args = (cgf.ell_min_mockdata, cgf.ell_max_mockdata, N_ELL)
    binned = {"kk": cgf.bin_ell_kk(*ell_args, Bs),
              "kg": cgf.bin_ell_delk(*ell_args, Bs, Bl),
              "gg": cgf.bin_ell_deldel(*ell_args, Bl)}
    pk2d = {"kk": Pk_kk, "kg": Pk_kg, "gg": Pk_gg}

    blocks = []
    for probe in ("kk", "kg", "gg"):
        Cl = cgf.Cell_CuGal(binned[probe], a_arr, UE_arr, coupling_arr, f_phi, cosmo, cgf.z, Bs, Bl,
                            bias, pk2d[probe], tracer1_type=probe[0], tracer2_type=probe[1])[1]
        blocks.append(np.array(Cl).flatten())
    return np.concatenate(blocks)


def gr_datavector(cosmo):
    """ GR 3x2pt vector at `cosmo`, ordered [kk, delk, deldel]. """
    Bs, Bl = cgf.Binned_distribution_source, cgf.Binned_distribution_lens
    ell_args = (cgf.ell_min_mockdata, cgf.ell_max_mockdata, N_ELL)
    binned = {"kk": cgf.bin_ell_kk(*ell_args, Bs),
              "kg": cgf.bin_ell_delk(*ell_args, Bs, Bl),
              "gg": cgf.bin_ell_deldel(*ell_args, Bl)}
    blocks = []
    for probe in ("kk", "kg", "gg"):
        Cl = cgf.Cell_GR(binned[probe], cosmo, cgf.z, Bs, Bl, cgf.Bias_distribution_fiducial,
                         tracer1_type=probe[0], tracer2_type=probe[1])[1]
        blocks.append(np.array(Cl).flatten())
    return np.concatenate(blocks)


########################## panel layout ##########################

def panel_layout():
    """ List of (grid row, grid col, probe, title, block index in the data vector, N_ELL entries each).
    The data vector is ordered [kk (i<=j), delk (lens j, source k), deldel]. """
    panels = []
    block = 0
    for a in range(1, 6):  # cosmic shear: upper-left triangle
        for b in range(a, 6):
            panels.append((a - 1, b - a, "kk", rf"$i_s,i_s=({a},{b})$", block))
            block += 1
    for j in range(5):  # clustering-lensing: same selection as C_ell_arr_delk_GR
        for k in range(5):
            if k - 1 > j or (k == 4 and j == 3):
                panels.append((5 - j, k + 1, "kg", rf"$i_l,i_s=({j + 1},{k + 1})$", block))
                block += 1
    for a in range(1, 6):  # clustering: anti-diagonal
        panels.append((6 - a, a, "gg", rf"$i_l,i_l=({a},{a})$", block))
        block += 1
    assert block * N_ELL == len(cgf.D_mockdata), "panel layout does not match the data-vector length"
    assert len({(p[0], p[1]) for p in panels}) == len(panels), "two panels share a grid cell"
    return panels


########################## compute ##########################

cosmo_u = cgf.cosmo_universe
f_phi_u = float(cgf.f_phi_universe)
theta_input = {"Omega_m": cosmo_u["Omega_m"], "f_phi": f_phi_u, "1e9As": cosmo_u["A_s"] * 1e9, "h": cosmo_u["h"],
               "ns": cosmo_u["n_s"], "Omega_b": cosmo_u["Omega_b"],
               **{f"b{i + 1}": cgf.Bias_distribution_fiducial[i][0] for i in range(5)}}

D_data = np.asarray(cgf.D_mockdata)
D_model = model_datavector(theta_input)
D_GR = gr_datavector(cosmo_u)  # NB: cgf.D_mockdata_GR is at cosmo_fid, not at the input universe
ell = np.asarray(cgf.ell_mockdata)
sigma = np.sqrt(np.diag(cgf.SRD_compare))
cut = np.all(cgf.gauss_invcov_rotated == 0, axis=0)

res_data = (D_data - D_GR) / D_GR
res_model = (D_model - D_GR) / D_GR
rel_err = sigma / D_GR

########################## checks ##########################

invcov = cgf.gauss_invcov_rotated
loglike_own = -0.5 * (D_data - D_model) @ invcov @ (D_data - D_model)
loglike_code = cgf.log_likelihood(theta_input, cgf.C_ell_data_mock, invcov)
rel_diff = np.abs(D_model - D_data) / np.abs(D_data)
n_cut = int(cut.sum())
print(f"data type {cgf.config['data']['type']}, f_phi = {f_phi_u:.4f}, {n_cut}/{len(cut)} points removed by scale cuts")
print(f"log-likelihood at input cosmology: {loglike_code:.6g} (recomputed from model vector: {loglike_own:.6g})")
for name, sl in (("kk", slice(0, 15 * N_ELL)), ("delk", slice(15 * N_ELL, 22 * N_ELL)), ("deldel", slice(22 * N_ELL, None))):
    keep = ~cut[sl]
    print(f"  {name:7s} max |model-data|/data = {rel_diff[sl].max():.3e} (uncut points: "
          f"{rel_diff[sl][keep].max() if keep.any() else np.nan:.3e}), "
          f"max |data residual| = {np.abs(res_data[sl]).max():.3e}")

########################## plot ##########################

col_list = matplotlib.colormaps["tab10"].colors
probe_colour = {"kk": col_list[0], "gg": col_list[1], "kg": col_list[3]}
model_colour, data_colour = col_list[2], "purple"

panels = panel_layout()
occupied = {(p[0], p[1]) for p in panels}

fig = plt.figure(figsize=(20, 16))
outer = gridspec.GridSpec(6, 6, wspace=0.35, hspace=0.3)

for row, col, probe, title, block in panels:
    sl = slice(block * N_ELL, (block + 1) * N_ELL)
    ax = fig.add_subplot(outer[row, col])
    ax.set_title(title, fontsize=19)
    for spine in ax.spines.values():
        spine.set_edgecolor(probe_colour[probe])
        spine.set_linewidth(2.5)

    x = ell[sl]
    ax.axhline(0, color="black", linewidth=0.8)
    ax.plot(x, res_model[sl], "-", color=model_colour, linewidth=2.2, zorder=3)
    ax.plot(x, res_data[sl], "--", color=data_colour, linewidth=1.8, zorder=4)

    # 1-sigma errors from the covariance
    ax.fill_between(x, -rel_err[sl], rel_err[sl], color="grey", alpha=0.3)
    ax.plot(x, rel_err[sl], color="grey", linewidth=0.8)
    ax.plot(x, -rel_err[sl], color="grey", linewidth=0.8)

    # shade the region removed by the scale cuts
    if cut[sl].any():
        ax.axvspan(x[cut[sl]].min(), args.xmax, color="red", alpha=0.08, linewidth=0)

    ax.set_xscale("log")
    ax.set_xlim(20, args.xmax)
    if col == 0:
        ax.set_ylabel(r"$(C_\ell - C_\ell^{\mathrm{GR}})/C_\ell^{\mathrm{GR}}$", fontsize=18)
    if (row + 1, col) in occupied:  # x tick labels only on the lowest panel of each column
        ax.tick_params(labelbottom=False)
    else:
        ax.set_xlabel(r"$\ell$", fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=16)

legend_elements = [
    Patch(facecolor="white", edgecolor=probe_colour["kk"], label="Cosmic Shear", linewidth=2.5),
    Patch(facecolor="white", edgecolor=probe_colour["gg"], label="Galaxy Clustering", linewidth=2.5),
    Patch(facecolor="white", edgecolor=probe_colour["kg"], label="Cross", linewidth=2.5),
    Line2D([0], [0], color=data_colour, linestyle="--", linewidth=2.5, label="Data vector"),
    Line2D([0], [0], color=model_colour, linestyle="-", linewidth=2.5, label="Model at input cosmology"),
    Patch(facecolor="grey", alpha=0.3, label=r"$1\sigma$ (diagonal covariance)"),
    Patch(facecolor="red", alpha=0.15, label="Removed by scale cuts"),
]
fig.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 0.97), ncol=4, frameon=False, fontsize=17)

# input-universe parameters in the empty top-right cell
ax_info = fig.add_subplot(outer[0, 5])
ax_info.axis("off")
ax_info.text(0, 1,
             rf"$f_\phi={f_phi_u:.3f}$" "\n"
             rf"$\Omega_m={cosmo_u['Omega_m']:.4f}$" "\n"
             rf"$h={cosmo_u['h']:.4f}$" "\n"
             rf"$n_s={cosmo_u['n_s']:.4f}$" "\n"
             rf"$A_s={cosmo_u['A_s']:.3e}$" "\n"
             f"data type {cgf.config['data']['type']}",
             fontsize=14, va="top", transform=ax_info.transAxes)

os.makedirs(os.path.dirname(out_path), exist_ok=True)
fig.savefig(out_path, bbox_inches="tight")
print("saved", out_path)
if args.show:
    plt.show()

# the recomputed likelihood must agree with the one the sampler uses
assert np.isclose(loglike_own, loglike_code, rtol=1e-6, atol=1e-8), \
    f"model vector does not reproduce log_likelihood: {loglike_own} vs {loglike_code}"
