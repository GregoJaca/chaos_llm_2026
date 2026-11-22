## Summary of Changes

This Pull Request brings the complete large-model campaign (conducted on the GB10 GPU `spark`) into the `master` branch. All four models requested for the large-model campaign are complete, validated, and documented:
1. **DeepSeek-R1-Distill-Qwen-14B** (`ds14`): 48 blocks, $d=5120$, true sliding window $W=100$ and full attention comparisons.
2. **Qwen/Qwen3.5-9B** (`qwen35`): 32 blocks, $d=4096$, hybrid Gated-DeltaNet/full attention architecture.
3. **microsoft/phi-4** (`phi4`): 40 blocks, $d=5120$, full attention architecture.
4. **Qwen/Qwen3-14B** (`qwen314`): 40 blocks, $d=5120$, full attention architecture.

### Key Scientific Takeaways
- **The 14B Fragility Comparison:** Comparing **Qwen3-14B vs DeepSeek-R1-Distill-Qwen-14B** (matched scale class, both $d=5120$) reveals that perturbation fragility is **family- and post-training-dependent, NOT scale-driven**. Qwen3-14B has a median flux tube radius $r_* \approx 8.39$ (0% flips at $r=0.1$, 0.2% at $r=0.3$) whereas DS-R1-14B has $r_* \approx 0.103$ (48.3% flips at $r=0.1$, ~70% at $r=0.3$), driven by a 46x thicker margin landscape and 2.8x lower decision sensitivity in Qwen3-14B.
- **Continuous Contraction:** In all large models, state-normalized perturbations $\|\delta h_\ell\| / \|h_\ell\|$ contract with depth (matching the findings from Mamba and the 1.5B small models in `master`).
- **Dimensionality ($D_2$):** Zero trajectories (0/49) across all large models display a valid power-law scaling window, demonstrating that low-dimensional chaotic attractor interpretations were artifacts of failing to enforce plateau tests. Point-cloud estimators (TwoNN, Levina-Bickel) consistently find intrinsic dimensionality $d \approx 6 - 8$.

### Namespace Separation (No Overlap with Small Model Work)
All large model additions have been placed into dedicated namespaces, ensuring zero breakage of existing code or results in `master`:
- Notebook & Figures: `notebook_big/` (`notebook_big.md`, `notebook_big.pdf`, HTML preview, figures, sections)
- Execution Engine: `src/chaos_llm/*_big.py` (`big_engine.py`, `run_a1_big.py`, `run_free_big.py`, `run_calib_big.py`, `run_depth_big.py`, `run_nonnorm_big.py`, `run_state2_big.py`, `collect_trajectories_big.py`)
- GPU & Analysis Scripts: `scripts_spark/`
- Output Data: `results_big/` (metadata, summary jsons, distance matrices)
- Documentation: `docs/SPARK_HANDOFF.md`
