ins = """![Fig. F7](update_figures/pack_loop_metric_M.png)

**Fig. F7 — mean log amplification.** $M=\\langle\\log(a_t/a_1)\\rangle_t$ over the whole run, median over directions, at $T=1$; bars are closed loop, replay, CRN and independent. **Data:** `results/softloop*` / `results/longloop*`, 300-3000 steps. **Read:** the closed loop has $M=1$-$7$ nats; replay and CRN sit at 0 or below; independent sampling is near 0. **Why keep it next to $L_\\infty$:** it integrates the whole trajectory, so it does not depend on where the curve saturates, but it is still horizon-dependent (the run length is reported).

"""
import io
P = "notebook/update.md"
s = io.open(P, encoding="utf-8").read()
assert "![Fig. F7b]" in s
s = s.replace("![Fig. F7b]", ins + "![Fig. F7b]", 1)
io.open(P, "w", encoding="utf-8").write(s)
print("inserted metric_M as F7")
