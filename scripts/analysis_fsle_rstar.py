"""Deep analysis of the FSLE and the flux-tube radius r*, with ~20 diagnostic plots."""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"C:\Users\hjaca\research\chaos_llm_2026"
LD = os.path.join(ROOT, "results", "local_data")
OUT = os.path.join(ROOT, "notebook", "figures", "fsle")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
                     "legend.fontsize": 8.5})
NUM = {}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=185)
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------------- load DeepSeek multi-radius data
s = np.load(os.path.join(ROOT, "results", "a1_ds_w100", "a1.npz"))
l = np.load(os.path.join(ROOT, "results", "a1_ds_w100_large", "a1.npz"))
radii = np.concatenate([s["radii"], l["radii"]])
D_ = np.concatenate([s["dh_norm"], l["dh_norm"]], 0)[:, :, -1, :]      # [nr, K, T]
_, idx = np.unique(np.round(radii, 8), return_index=True)
radii, D_ = radii[idx], D_[idx]
T = D_.shape[-1]
t = np.arange(T)
NUM["radii"] = radii.tolist()


def tau_of(x, rho, t0=0):
    if x[t0] <= 0:
        return None
    hit = np.where(x[t0:] >= rho * x[t0])[0]
    return int(hit[0]) if len(hit) else None


# ---------------------------------------------------------------- 1. normalised curves (the object)
fig, ax = plt.subplots(figsize=(8.6, 4.8))
for r, c in [(1e-3, "0.65"), (1e-2, "tab:blue"), (0.1, "tab:orange"), (0.3, "tab:green")]:
    i = int(np.argmin(np.abs(radii - r)))
    xn = D_[i] / np.maximum(D_[i][:, :1], 1e-30)
    med = np.median(xn, axis=0)
    ax.semilogy(t, np.maximum(med, 1e-3), lw=1.1, color=c, label=f"$r={r}$")
ax.axhline(2, color="r", ls="--", lw=0.9)
ax.set_xlabel("generated token $t$")
ax.set_ylabel(r"$\|\delta h_t\|\,/\,\|\delta h_0\|$")
ax.set_title("DeepSeek: the normalised response is the same at every radius\n"
             r"(linear response; $\times2$ line used by the FSLE)")
ax.legend(); ax.grid(alpha=0.25, which="both")
save(fig, "fsle_norm_curves.png")

# collapse quality
xn = {}
for r in [1e-3, 0.01, 0.1, 0.3]:
    i = int(np.argmin(np.abs(radii - r)))
    xn[r] = np.median(D_[i] / np.maximum(D_[i][:, :1], 1e-30), axis=0)
ref = xn[0.1]
NUM["collapse_max_rel_dev"] = {str(r): float(np.nanmax(np.abs(v[:200] - ref[:200]) / np.maximum(ref[:200], 1e-6)))
                               for r, v in xn.items()}

# ---------------------------------------------------------------- 2. tau(rho) and lambda(rho)
rhos = np.array([1.2, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0])
i01 = int(np.argmin(np.abs(radii - 0.1)))
tau = np.array([np.median([q for q in (tau_of(D_[i01, di], rho) for di in range(D_.shape[1])) if q])
                for rho in rhos])
lam = np.log(rhos) / tau
NUM["tau_rho"] = dict(rho=rhos.tolist(), tau=tau.tolist(), lam=lam.tolist())

fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.5))
ax = axs[0]
ax.plot(rhos, tau, "o-", color="tab:blue")
ax.set_xlabel(r"growth threshold $\varrho$")
ax.set_ylabel(r"median $\tau(\varrho)$ (tokens)")
ax.set_title(r"(a) Time to grow by $\varrho$ at $r=0.1$")
ax.grid(alpha=0.25)
ax = axs[1]
ax.plot(rhos, lam, "o-", color="tab:red")
ax.set_xlabel(r"growth threshold $\varrho$")
ax.set_ylabel(r"$\lambda_{\rm FSLE}=\log\varrho/\tau(\varrho)$")
ax.set_title("(b) FSLE as a function of the threshold\n(not a single number)")
ax.grid(alpha=0.25)
save(fig, "fsle_tau_and_lambda_vs_rho.png")

# ---------------------------------------------------------------- 3. lambda vs r, per rho
fig, axs = plt.subplots(1, 2, figsize=(12.6, 4.6))
ax = axs[0]
for rho, c in [(1.5, "tab:blue"), (2.0, "tab:orange"), (4.0, "tab:green")]:
    y, cen = [], []
    for i in range(len(radii)):
        q = [tau_of(D_[i, di], rho) for di in range(D_.shape[1])]
        ok = [x for x in q if x]
        y.append(np.log(rho) / np.median(ok) if ok else np.nan)
        cen.append(1 - len(ok) / len(q))
    ax.plot(radii, y, "o-", color=c, label=f"$\\varrho={rho}$")
ax.set_xscale("log"); ax.set_xlabel(r"radius $r$")
ax.set_ylabel(r"FSLE $\lambda$")
ax.set_title("(a) FSLE is flat in $r$ (except at the floor)")
ax.legend(); ax.grid(alpha=0.25, which="both")
ax = axs[1]
for rho, c in [(1.5, "tab:blue"), (2.0, "tab:orange"), (4.0, "tab:green")]:
    cen = []
    for i in range(len(radii)):
        q = [tau_of(D_[i, di], rho) for di in range(D_.shape[1])]
        cen.append(1 - sum(1 for x in q if x) / len(q))
    ax.plot(radii, cen, "o-", color=c, label=f"$\\varrho={rho}$")
ax.set_xscale("log"); ax.set_xlabel(r"radius $r$")
ax.set_ylabel("fraction censored")
ax.set_title(r"(b) Censoring: the $\varrho=4$ curve is only defined where $r\gtrsim10^{-3}$")
ax.legend(); ax.grid(alpha=0.25, which="both")
save(fig, "fsle_vs_r_censoring.png")

# ---------------------------------------------------------------- 4. two references
for t0 in (0, 20):
    y = []
    for i in range(len(radii)):
        q = [tau_of(D_[i, di], 2.0, t0) for di in range(D_.shape[1])]
        ok = [x for x in q if x and x > 0]
        y.append(np.log(2) / np.median(ok) if ok else np.nan)
    if t0 == 0:
        y0 = y
    else:
        y20 = y
fig, ax = plt.subplots(figsize=(8.2, 4.5))
ax.plot(radii, y0, "o-", label=r"reference $t_0=0$")
ax.plot(radii, y20, "s-", label=r"reference $t_0=20$")
ax.set_xscale("log"); ax.set_xlabel(r"radius $r$"); ax.set_ylabel(r"FSLE $\lambda$ ($\varrho=2$)")
ax.set_title("The FSLE value depends on where the clock starts\n"
             "(the response is a transient, not a growth law)")
ax.legend(); ax.grid(alpha=0.25, which="both")
save(fig, "fsle_reference_choice.png")

# ---------------------------------------------------------------- 5. tau distribution
i = int(np.argmin(np.abs(radii - 0.1)))
taus = np.array([q for q in (tau_of(D_[i, di], 2.0) for di in range(D_.shape[1])) if q])
fig, ax = plt.subplots(figsize=(7.6, 4.4))
ax.hist(taus, bins=np.arange(-0.5, taus.max() + 1.5), color="tab:blue", alpha=0.8)
ax.axvline(np.median(taus), color="r", ls="--", label=f"median {np.median(taus):.0f}")
ax.set_xlabel(r"$\tau$ (tokens) to double")
ax.set_ylabel("directions")
ax.set_title(f"Doubling times are quantised and concentrated ($r=0.1$, n={len(taus)})")
ax.legend(); ax.grid(alpha=0.25)
save(fig, "fsle_tau_hist.png")

# ---------------------------------------------------------------- 6. FSLE vs lambda_10
k = 10
a = np.maximum(np.median(D_[i], axis=0), 1e-12)
tt = t[k:-k]
lam10 = np.log(a[tt + k] / a[tt - k]) / (2 * k)
fig, ax = plt.subplots(figsize=(7.6, 4.5))
ax.plot(tt, lam10, lw=1.2, color="tab:red", label=r"$\lambda_{10}(t)$")
ax.axhline(np.log(2) / np.median(taus), color="tab:blue", ls="--",
           label=r"FSLE ($\varrho=2$, global)")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("generated token $t$"); ax.set_ylabel("rate (nats/token)")
ax.set_title("FSLE is an average over the whole transient;\n$\\lambda_{10}(t)$ is local and changes sign")
ax.legend(); ax.grid(alpha=0.25)
save(fig, "fsle_vs_lambda10.png")
NUM["fsle_global"] = float(np.log(2) / np.median(taus))
NUM["lam10_mean"] = float(np.mean(lam10))

# ---------------------------------------------------------------- 7. r* from r_c
d = np.load(os.path.join(LD, "w100_layers", "a1.npz"))
rc = d["r_c"].astype(float)
rstar = rc[0]                                   # radius index 0, all directions
rs = np.sort(rstar.min(axis=1))
rs = rs[np.isfinite(rs) & (rs > 0)]
ccdf = 1 - np.arange(len(rs)) / len(rs)
NUM["rstar"] = rs.tolist()

fig, axs = plt.subplots(1, 2, figsize=(12.6, 4.5))
ax = axs[0]
ax.loglog(rs, ccdf, "o-", ms=4, color="tab:blue")
ax.set_xlabel(r"$r_*$ (minimum radius that flips any token)")
ax.set_ylabel(r"$P(r_* > x)$")
ax.set_title("(a) CCDF of the flux-tube radius")
ax.grid(alpha=0.25, which="both")
ax = axs[1]
ax.hist(np.log10(rs), bins=12, color="tab:blue", alpha=0.8)
ax.set_xlabel(r"$\log_{10} r_*$"); ax.set_ylabel("directions")
ax.set_title("(b) Distribution of $r_*$ (20 directions)")
ax.grid(alpha=0.25)
save(fig, "rstar_ccdf_hist.png")

# r* per direction, sorted, with the flip at r=1.5 from the free runs if present
order = np.argsort(rstar.min(axis=1))
fig, ax = plt.subplots(figsize=(8.4, 4.4))
ax.semilogy(np.arange(len(order)), np.sort(rstar.min(axis=1)), "o-", color="tab:blue")
ax.axhline(0.1, color="0.6", ls=":", label=r"$r=0.1$")
ax.axhline(0.3, color="0.4", ls=":", label=r"$r=0.3$")
ax.set_xlabel("direction (sorted)")
ax.set_ylabel(r"$r_*$")
ax.set_title("Every direction has a finite flip radius;\nthe spread is a factor 5")
ax.legend(); ax.grid(alpha=0.25, which="both")
save(fig, "rstar_per_direction.png")
NUM["rstar_spread"] = float(np.max(rs) / np.min(rs))

# local r_c(t) for the 5 most fragile directions
fig, ax = plt.subplots(figsize=(8.6, 4.4))
for j in range(5):
    di = order[j]
    ax.semilogy(t[:160], np.maximum(rstar[di][:160], 1e-4), lw=1.0, label=f"dir {di}")
ax.set_ylim(1e-4, 10)
ax.set_xlabel("generated token $t$"); ax.set_ylabel(r"$r_c(t,u)$")
ax.set_title(r"Local flip radius $r_c(t,u)$ for the five most fragile directions")
ax.legend(fontsize=8); ax.grid(alpha=0.25, which="both")
save(fig, "rstar_local_curves.png")

# r* vs the margin at the argmin
i_min = np.argmin(rstar, axis=1)
m_at = d["margins"][i_min]
fig, ax = plt.subplots(figsize=(7.4, 4.4))
ax.loglog(rs, np.sort(m_at)[np.argsort(np.argsort(rs))] if False else m_at[np.argsort(rstar.min(axis=1))], "o")
ax.set_xlabel(r"$r_*$"); ax.set_ylabel("margin at the argmin")
ax.set_title(r"$r_*$ tracks the decision margin at the closest call")
ax.grid(alpha=0.25, which="both")
save(fig, "rstar_vs_margin.png")
ix = np.argsort(rstar.min(axis=1))
NUM["corr_rstar_margin"] = float(np.corrcoef(np.log(rs), np.log(np.maximum(m_at[ix], 1e-6)))[0, 1])

# ---------------------------------------------------------------- 8. cross-model r*
MODELS = [("DeepSeek-1.5B", os.path.join(LD, "big", "all_layers", "a1.npz"), 0.1),
          ("Qwen3-4B", os.path.join(LD, "models", "qwen3_4b", "a1.npz"), 0.1),
          ("Phi-4-mini", os.path.join(LD, "models", "phi", "a1.npz"), 0.3),
          ("Qwen3.5-2B", os.path.join(LD, "models", "qwen35", "a1.npz"), 0.1)]
fig, ax = plt.subplots(figsize=(8.4, 4.6))
for name, path, r in MODELS:
    if not os.path.exists(path):
        continue
    z = np.load(path, allow_pickle=True)
    if "r_c" not in z:
        continue
    ri = int(np.argmin(np.abs(z["radii"] - r)))
    rr = np.sort(z["r_c"][ri].min(axis=1))
    rr = rr[np.isfinite(rr) & (rr > 0)]
    ax.loglog(rr, 1 - np.arange(len(rr)) / len(rr), "o-", ms=4, label=f"{name} ($r={r}$)")
ax.set_xlabel(r"$r_*$"); ax.set_ylabel("CCDF")
ax.set_title("Flux-tube radius across models")
ax.legend(); ax.grid(alpha=0.25, which="both")
save(fig, "rstar_models.png")

json.dump(NUM, open(os.path.join(ROOT, "results", "fsle_rstar_numbers.json"), "w"), indent=1)
print("done")
