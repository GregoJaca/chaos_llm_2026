"""Part 4 figures: state-level Jacobian measurements for Mamba (negative control).

Data: in-batch finite-difference Jacobian runs (run_mamba_jacobian.py):
  A  state-kick growth curves (perturb the full state at t0, teacher-force H tokens)
  C  state-Jacobian diagonal spectrum (log exp(dt*A) per layer/channel/step)
  B  one-step sigma_max by power iteration + transient amplification
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CTRL = os.path.join(ROOT, "results", "ctrl_local", "mamba130")
OUT = os.path.join(ROOT, "notebook", "figures_controls")
os.makedirs(OUT, exist_ok=True)

C130 = "#0072B2"
C370 = "#56B4E9"

z130 = np.load(os.path.join(CTRL, "jac130.npz"), allow_pickle=True)
z370 = np.load(os.path.join(CTRL, "jac370.npz"), allow_pickle=True)

# ---------------- A: state-kick growth ----------------
fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
for ax, z, c, name in [(axs[0], z130, C130, "Mamba-130m"), (axs[1], z370, C370, "Mamba-370m")]:
    dstate = z["A_dstate"].item()
    for t0 in sorted(dstate.keys(), key=int):
        D = dstate[t0]  # [K, H]
        med = np.median(D, axis=0)
        ax.plot(np.arange(len(med)), med, lw=1.2, color=c, label=f"$t_0$={t0}")
        ax.fill_between(np.arange(len(med)), np.quantile(D, .25, axis=0),
                        np.quantile(D, .75, axis=0), color=c, alpha=0.12)
    ax.axhline(1.0, color="k", lw=0.6, ls=":")
    ax.set_xlabel(r"tokens after the state kick")
    ax.set_ylabel(r"$\|\delta \mathrm{state}(t)\|$  (all layers)")
    ax.set_title(f"{name}: kick $\\varepsilon=10^{{-2}}$ at $t_0$,\nteacher-forced -- bounded, no growth")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
ax = axs[2]
d130 = z130["A_dstate"].item()
d370 = z370["A_dstate"].item()
for t0 in sorted(d130.keys(), key=int):
    m130 = np.median(d130[t0], axis=0)
    m370 = np.median(d370[t0], axis=0)
    ax.plot(np.arange(len(m130)), m130 / m130[0], color=C130, lw=1.0)
    ax.plot(np.arange(len(m370)), m370 / m370[0], color=C370, lw=1.0, ls="--")
ax.axhline(1.0, color="k", lw=0.6, ls=":")
ax.set_xlabel("tokens after the state kick")
ax.set_ylabel(r"$\|\delta\mathrm{state}(t)\|/\|\delta\mathrm{state}(0)\|$ (median)")
ax.set_title("Normalised: a transient bump (x1.5-3, non-normality)\nthen flat or decaying -- never exponential growth")
ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "c4_kick.png"), dpi=190); plt.close(fig)

# ---------------- C: exponent spectrum ----------------
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))
ax = axs[0]
pooled = {}
for z, c, name in [(z130, C130, "130m"), (z370, C370, "370m")]:
    C = z["C_logdiag"]  # [nt, nl, per_t, d_state]
    vals = C.flatten()
    vals = vals[np.isfinite(vals)]
    pooled[name] = vals
    hist, edges = np.histogram(vals, bins=80, range=(-8, 0.05), density=True)
    ax.plot(0.5 * (edges[1:] + edges[:-1]), hist, lw=1.2, color=c,
            label=f"{name}: med {np.median(vals):.2f}")
ax.axvline(0, color="k", lw=1)
ax.set_xlabel(r"$\log J_{ii} = \Delta_t A_i$  (per-step state-exponent sample)")
ax.set_ylabel("density (all layers, channels, steps pooled)")
ax.set_title("(a) State-Jacobian diagonal: every sampled exponent is negative;\nlong slow-channel tail near 0 carries the response (Part 2)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax = axs[1]
for z, c, name in [(z130, C130, "130m"), (z370, C370, "370m")]:
    C = z["C_logdiag"]
    layers = z["C_layers"]
    nt = C.shape[0]
    med = np.median(C.reshape(nt, -1), axis=1)
    ax.plot(z["C_t"], med, "o-", color=c, label=f"{name} (median over layers)")
    q10 = np.quantile(C.reshape(nt, -1), 0.9, axis=1)  # 90th pct = slowest tail
    ax.plot(z["C_t"], q10, "s--", color=c, alpha=0.6, label=f"{name} 90th pct (slowest)")
ax.axhline(0, color="k", lw=1)
ax.set_xlabel("token $t$")
ax.set_ylabel(r"$\log J_{ii}$")
ax.set_title("(b) Slowest channels per step: exponents near $-10^{-2}$ to $-10^{-3}$\n= state half-lives of hundreds of tokens")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "c4_diag.png"), dpi=190); plt.close(fig)

# ---------------- B: sigma_max + transient ----------------
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))
ax = axs[0]
for z, c, name in [(z130, C130, "130m"), (z370, C370, "370m")]:
    ax.plot(z["C_t"], z["B_sigma"], "o-", color=c, label=name)
    chain = z["B_chain"]  # [nt, 16]
    ax.errorbar(z["C_t"], z["B_sigma"], yerr=chain.std(1) * 0 + 0.02, color=c, alpha=0.4)
ax.axhline(1.0, color="k", lw=1)
ax.set_xlabel("token $t$")
ax.set_ylabel(r"$\sigma_{\max}(J_t)$ (power iteration)")
ax.set_title(r"(a) One-step $\sigma_{\max}\approx 0.99$-1.01: the worst direction is\nmarginally non-expanding; all eigenvalues $<1$")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax = axs[1]
for z, c, name in [(z130, C130, "130m"), (z370, C370, "370m")]:
    chain = z["B_chain"]  # [nt, niters]
    for ti in range(chain.shape[0]):
        ax.plot(np.arange(1, chain.shape[1] + 1), chain[ti], color=c, lw=0.9, alpha=0.6)
ax.axhline(0, color="k", lw=1)
ax.set_xlabel("power-iteration number")
ax.set_ylabel(r"$\log \|J_t v\|$ per iteration")
ax.set_title("(b) Random-vector response: first application amplifies ($\\log\\|Jv\\|$ up to 2.3,\ni.e. x10 transient, non-normality), then contracts")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "c4_sigma.png"), dpi=190); plt.close(fig)

# numbers for the text
print("leak (max |response| in layers below the perturbed layer):",
      float(np.max(z130["C_below"])), float(np.max(z370["C_below"])))
for z, name in [(z130, "130m"), (z370, "370m")]:
    C = z["C_logdiag"].flatten()
    C = C[np.isfinite(C)]
    print(f"{name}: log J_ii median {np.median(C):.3f}, 90th pct {np.quantile(C, 0.9):.4f}, "
          f"99th {np.quantile(C, 0.99):.4f}, max {C.max():.4f}")
    print(f"  sigma_max: {np.round(z['B_sigma'], 3)}, transient first-iter max {z['B_chain'][:, 0].max():.2f}")
    D = z["A_dstate"].item()
    for t0 in sorted(D.keys(), key=int):
        m = np.median(D[t0], axis=0)
        print(f"  A t0={t0}: {m[0]:.4g} -> +50 {m[50]:.4g} -> end {m[-1]:.4g} (peak {m.max():.4g} at {m.argmax()})")
print("wrote", OUT)
