import os, sys, glob, re, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import *
from style import plt, save

D_HID = {"deepseek": 1536, "qwen3_4b": 2560, "phi4mini": 3072, "qwen25_3b": 2048, "qwen3_5": 2048}

def load_series(dirpat, D):
    ps = sorted(glob.glob(os.path.join(HERE, dirpat)), key=lambda p: int(re.search(r"w(\d+)", p).group(1)))
    Ws = np.array([int(re.search(r"w(\d+)", p).group(1)) for p in ps])
    ref = np.load(ps[-1], allow_pickle=True)
    P = len(ref["prompt_ids"])
    p_last = int(np.max(ref["subspace"]) // D)
    FL = []
    for p in ps:
        z = np.load(p, allow_pickle=True)
        a = z["dh_norm"][0].max(axis=0)      # (layer, token)
        fl = []
        for l in range(a.shape[0]):
            nz = np.where(a[l] > 1e-7)[0]
            fl.append(int(nz.max() + 1) if len(nz) else 0)
        FL.append(fl)
    return Ws, np.array(FL), P, p_last, ref["layers"].astype(int)

Ws, FL, P, p_last, layers = load_series("results/local_data/wspec/w*/a1.npz", D_HID["deepseek"])
print("P", P, "p_last", p_last, "W", Ws.min(), Ws.max())

fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8))
ax = axes[0]
li = 0
ok = (FL[:, li] > 0) & (FL[:, li] < 199)
pred = p_last + Ws - P
ax.plot(Ws[ok], FL[ok, li], ".", ms=5, color=COL["DeepSeek-1.5B"], label="measured")
ax.plot(Ws[ok], pred[ok], "k-", lw=1.0, label=r"$p_{\rm last}+W-P$")
ax.set_xlabel("window size $W$")
ax.set_ylabel(r"flush step $t^*_1$")
ax.set_title(f"(a) 97 windows; $p_{{\\rm last}}$={p_last}, $P$={P}", loc="left")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1]
for W in [60, 80, 100, 120, 150, 200]:
    i = int(np.argmin(np.abs(Ws - W)))
    fl = FL[i].astype(float)
    predline = Ws[i] - P + p_last + (layers - 1) * (Ws[i] - 1)
    ok2 = (fl > 0) & (fl < 200) & (predline < 210)
    ax.plot(layers[ok2], fl[ok2], "o", ms=4, label=f"W={Ws[i]}")
    ax.plot(layers[ok2], predline[ok2], "-", lw=0.9, alpha=0.7)
ax.set_xlabel("layer index $\\ell$")
ax.set_ylabel(r"$t^*_\ell$")
ax.set_title(r"(b) $t^*_\ell=W-P+p_{\rm last}+(\ell-1)(W-1)$", loc="left")
ax.set_ylim(0, 210)
ax.legend(fontsize=6.6); ax.grid(alpha=0.3)

ax = axes[2]
for li, l in enumerate(layers):
    ok = (FL[:, li] > 0) & (FL[:, li] < 200) & ((Ws - P + p_last + (l - 1) * (Ws - 1)) < 210)
    if ok.sum() < 3:
        continue
    resid = FL[ok, li] - (Ws[ok] - P + p_last + (l - 1) * (Ws[ok] - 1))
    ax.plot(Ws[ok], resid, ".", ms=4, label=f"layer {l}")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("window size $W$")
ax.set_ylabel("residual (tokens)")
ax.set_title("(c) Residual from the exact formula", loc="left")
ax.legend(fontsize=6.4, ncol=2); ax.grid(alpha=0.3)
fig.tight_layout()
save(fig, "pack_window_exact.png")

# model comparison
fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
ax = axes[0]
for m, D, c in [("qwen3_4b", 2560, COL["Qwen3-4B"]), ("phi4mini", 3072, COL["Phi-4-mini"])]:
    ps = sorted(glob.glob(os.path.join(HERE, f"results/local_data/window4d/{m}/w*/a1.npz")),
                key=lambda p: int(re.search(r"w(\d+)", p).group(1)))
    Ws2 = np.array([int(re.search(r"w(\d+)", p).group(1)) for p in ps])
    ref = np.load(ps[-1], allow_pickle=True); P2 = len(ref["prompt_ids"]); pl = int(np.max(ref["subspace"]) // D)
    fl1 = []
    for p in ps:
        z = np.load(p, allow_pickle=True); a = z["dh_norm"][0].max(axis=0)
        nz = np.where(a[0] > 1e-7)[0]; fl1.append(int(nz.max() + 1) if len(nz) else 0)
    ax.plot(Ws2, fl1, "o", color=c, label=f"{m}: p_last={pl}, P={P2}")
    ax.plot(Ws2, pl + Ws2 - P2, "-", color=c, alpha=0.7)
ax.set_xlabel("$W$"); ax.set_ylabel(r"$t^*_1$")
ax.set_title("(a) The same formula on Qwen3-4B / Phi-4-mini", loc="left")
ax.legend(fontsize=7.4); ax.grid(alpha=0.3)
ax = axes[1]
for m, D, c in [("qwen3_4b", 2560, COL["Qwen3-4B"]), ("phi4mini", 3072, COL["Phi-4-mini"])]:
    ps = sorted(glob.glob(os.path.join(HERE, f"results/local_data/window4d/{m}/w*/a1.npz")),
                key=lambda p: int(re.search(r"w(\d+)", p).group(1)))
    ref = np.load(ps[-1], allow_pickle=True); P2 = len(ref["prompt_ids"]); pl = int(np.max(ref["subspace"]) // D)
    for p in ps[:3]:
        z = np.load(p, allow_pickle=True); a = z["dh_norm"][0].max(axis=0)
        W = int(re.search(r"w(\d+)", p).group(1)); fl = []
        for l in range(a.shape[0]):
            nz = np.where(a[l] > 1e-7)[0]; fl.append(int(nz.max() + 1) if len(nz) else 0)
        fl = np.array(fl, float); fl[fl >= 300] = np.nan
        ax.plot(z["layers"], fl, "o", color=c, alpha=0.8)
        ax.plot(z["layers"], W - P2 + pl + (z["layers"] - 1) * (W - 1), "-", color=c, alpha=0.6)
ax.set_ylim(0, 300)
ax.set_xlabel("layer $\\ell$"); ax.set_ylabel(r"$t^*_\ell$")
ax.set_title("(b) Layer law, both models", loc="left")
ax.grid(alpha=0.3)
fig.tight_layout()
save(fig, "pack_window_exact_models.png")

json.dump(dict(P=int(P), p_last=int(p_last), Ws=Ws.tolist()), open(os.path.join(HERE, "analysis_update", "pack_window_exact.json"), "w"), indent=1)
print("done")
