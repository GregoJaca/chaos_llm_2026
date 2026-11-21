p = r"C:\Users\hjaca\research\chaos_llm_2026\scripts\analysis_kernel_models.py"
s = open(p, encoding="utf-8").read()

s = s.replace("""        amp = K.max(axis=1)
        As.append(amp)
    Bm = np.median(np.array(Bs), axis=0)
    Am = np.median(np.array(As), axis=0)""",
"""        amp = K.max(axis=1)
        pg = np.linspace(0, 1, 25)
        As.append(np.interp(pg, pos / pos[-1], amp))
    Bm = np.median(np.array(Bs), axis=0)
    Am = np.median(np.array(As), axis=0)
    pos = np.linspace(0, 1, 25)""")

s = s.replace("""    rows[tag] = dict(bg_start=float(Bm[0]), bg_end=float(Bm[-1]),
                     bg_ratio=float(Bm[-1] / max(Bm[0], 1e-9)),
                     amp_min=float(Am.min()), amp_max=float(Am.max()),
                     amp_argmax=int(pos[int(np.argmax(Am))]))""",
"""    rows[tag] = dict(bg_start=float(Bm[0]), bg_end=float(Bm[-1]),
                     bg_ratio=float(Bm[-1] / max(Bm[0], 1e-9)),
                     amp_min=float(Am.min()), amp_max=float(Am.max()),
                     amp_argmax=float(pos[int(np.argmax(Am))]))""")

s = s.replace('axs[1].set_xlabel("perturbed prompt position $p$")',
              'axs[1].set_xlabel("relative prompt position $p/P$")')
s = s.replace("""    z = np.load(files[0])
    K = np.median(z["dh"][:, :, -1, :], axis=1)
    pos = z["positions"]
    ax.plot(pos, np.argmax(K, axis=1), lw=1.2, color=c, label=NAME[tag])""",
"""    z = np.load(files[0])
    K = np.median(z["dh"][:, :, -1, :], axis=1)
    pos = z["positions"]
    ax.plot(pos / pos[-1], np.argmax(K, axis=1), lw=1.2, color=c, label=NAME[tag])""")
s = s.replace('ax.set_xlabel("perturbed prompt position $p$")',
              'ax.set_xlabel("relative prompt position $p/P$")')
open(p, "w", encoding="utf-8").write(s)
print("patched")
