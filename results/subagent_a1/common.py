"""Shared loader / helpers for A1 perturbation-response analysis (CPU, local only).

Data: chaos_llm_2026/results/local_data/{w100_layers,w200,w500}/a1.npz
  dh_norm[r, dir, layer, t] : L2 deviation of hidden state at layer output,
      in units of the embedding norm at the injection position (~1.22).
      (Verified: 1-cos^2 ~ (dh*1.217/norm_h0)^2 to <1e-4.)
  radii  : relative perturbation sizes r (12 values, 1e-3 .. 2.0)
  layers : measured layers; layer 0 is identically 0 (injection is downstream
      of the embedding output, so layer 0 never responds).
  margins[t] : base-run top1-top2 logit margin of the generated token stream.
  pred_T[r, dir] : first t where the perturbed run's next-token prediction
      differs from base (-1 = never).
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = r"C:\Users\hjaca\research\chaos_llm_2026\results"
OUT = ROOT + r"\subagent_a1"
DATA = {
    "w100": ROOT + r"\local_data\w100_layers\a1.npz",
    "w200": ROOT + r"\local_data\w200\a1.npz",
    "w500": ROOT + r"\local_data\w500\a1.npz",
}

plt.rcParams.update({
    "font.size": 10.5,
    "axes.titlesize": 11,
    "axes.labelsize": 10.5,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 110,
    "savefig.dpi": 170,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "legend.framealpha": 0.85,
})

# directions used for "3 individual + 1 aggregate" panels: typical / weakest / strongest
DIRS_SHOW = [0, 6, 10]


def load(w="w100"):
    d = np.load(DATA[w], allow_pickle=True)
    dh = d["dh_norm"].astype(np.float64)          # [r, dir, l, t]
    radii = d["radii"].astype(float)
    layers = d["layers"].astype(int)
    margins = d["margins"].astype(float)
    norm_h0 = d["norm_h0"].astype(float)          # [l, t] base hidden norms
    pred_T = d["pred_T"]
    dm = d["dm"].astype(float)
    dhm = np.median(dh, axis=1)                   # [r, l, t] median over directions
    iqr = np.percentile(dh, 75, axis=1) - np.percentile(dh, 25, axis=1)

    # empirical float32 noise floor per layer: smallest nonzero dh anywhere
    # (for shallow layers, restrict to pre-flush times; their late-t zeros are
    # a genuine bitwise flush, not noise)
    n_l = dh.shape[2]
    floor = np.zeros(n_l)
    for li in range(n_l):
        x = dh[:, :, li, :]
        if layers[li] <= 3:
            x = x[:, :, :80]
        pos = x[x > 0]
        floor[li] = pos.min() if pos.size else np.nan
    valid = dh > 3.0 * floor[None, None, :, None]  # [r, dir, l, t] bool

    small = radii <= 0.31                          # linear-regime radii (7)
    safe = (radii >= 0.03) & (radii <= 0.31)       # above noise floor at all t (4)
    large = radii >= 0.5                            # nonlinear-regime radii (5)

    # "universal curve" G(t,l) = median over safe radii of dh/r  (floor-free)
    G = np.full((dh.shape[2], dh.shape[3]), np.nan)
    nG = np.zeros((dh.shape[2], dh.shape[3]), int)
    for li in range(n_l):
        for t in range(dh.shape[3]):
            vals = [np.median(dh[ir, :, li, t]) / radii[ir] for ir in np.where(safe)[0]]
            G[li, t] = np.median(vals)
            nG[li, t] = len(vals)

    # flush time per layer: first t from which the median curve is exactly 0
    flush = {}
    for li in range(n_l):
        z = np.where(dhm[:, li, :].max(axis=0) == 0)[0]  # all radii zero
        if len(z) and z[-1] - z[0] + 1 == len(z) and z[0] > 10:
            flush[int(layers[li])] = int(z[0])

    return dict(d=d, dh=dh, dhm=dhm, iqr=iqr, radii=radii, layers=layers,
                margins=margins, norm_h0=norm_h0, pred_T=pred_T, dm=dm,
                floor=floor, valid=valid, small=small, safe=safe, large=large,
                G=G, nG=nG, flush=flush,
                gen_ids=d["gen_ids"], simplex=d["simplex"], subspace=d["subspace"])


def rcolor(r, radii):
    """color for a radius, viridis over log10 r."""
    lo, hi = np.log10(radii.min()), np.log10(radii.max())
    f = (np.log10(r) - lo) / (hi - lo)
    return plt.cm.viridis(0.12 + 0.76 * f)


def lcolor(l, layers):
    f = (np.log10(max(l, 1)) - 0) / np.log10(max(layers.max(), 2))
    return plt.cm.magma(0.18 + 0.66 * f)


def save(fig, name):
    fig.savefig(rf"{OUT}\{name}", bbox_inches="tight")
    print("saved", name)
    plt.close(fig)


def txt(name, s):
    with open(rf"{OUT}\{name}", "w") as f:
        f.write(s)
