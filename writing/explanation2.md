![Fig. C3](update_figures/pack_rotation_controls.png)

Top singular vectors of consecutive block Jacobians are close to orthogonal, but more aligned than random in trained LLMs. This explains why while each layer Jacobian has the *capacity* to amplify perturbations, the growth is limited.
// Careful with "close to orthogonal": 78-87 degrees is far from orthogonal in absolute terms; it is the *random control* that sits at 89-90. The correct statement is that trained models rotate less than random, and the consequence is that a fixed vector cannot follow the rotating optimal direction, so the product of the per-block maxima overestimates the true gain by orders of magnitude (cum_l is the honest object, Fig. B1).
// Also: a single top singular vector is fragile when two singular values are nearly equal (if they swap, the angle jumps). The k=3/k=10 subspace version (pack_topk.png, running now) is the robust version; expect the angles to move further from 90 for larger k.

![Fig. D3b](update_figures/pack_time_layers_controls.png)
clearly understand how teacher forced generation removes the "discrete sampling" (we don't have a "discrete" "step-function-like" increase) compared to what we have in autoregressive generation.
In general we see that there might be a transient growth but eventually perturbations decay.
We see clear differences of the response at different layers.
In most cases,
// "in most cases" was left unfinished: deeper layers retain more of the perturbation than shallow ones, in every trained model AND in Mamba; the random transformer is flat. Quantified in Fig. B6 (last/middle 3.5-14 for trained models, 1.2-1.7 for Pythia/random).
note: layer 1 is the *second* layer. layer 0 is always just a column of the vocabulary matrix, and thus constant (unless we perturb it)
// Confirmed by measurement: the layer-0 perturbation is EXACTLY zero for every generated token (all directions, all radii, all models) because the token embedding is a lookup. The perturbation can only enter at block 1, through attention. In the closed loop this changes (the fed embedding is a mixture of the perturbed logits), and there the layer-0 perturbation is nonzero - that is the mechanism, measured by the queued layer-0 runs.
This figure answers: Is the "discrete" "step-function-like" divergence you saw during autoregressive generation caused by the model itself or by the sampling (at each generation step). This answers: it was caused by sampling.
This is effectively a control that removes the sampling (because it samples the same as the teacher).
// Precise wording: teacher forcing feeds every row the *same* hard token stream, so it removes token divergence (the "discrete" jumps) but it also removes the mean feedback path. So it separates "the network with a frozen input stream" from "the network feeding back its own distribution"; it is not only a sampling control. The sampling-only control is CRN vs independent (Fig. F5).

![Fig. F0d](update_figures/pack_loop_loglog_ds15.png)
Following up on the previous figure (update_figures/pack_time_layers_controls.png) what we do now is looping.
We compare the "closed loop" against the "network only replay" (control) and we see that the closed loop amplifies perturbations as it generates tokens. eventually it saturates. The "replay" stays roughly constant at ~1, which means that the perturbation doesn't get amplified. interestingly it doesn't decay (as we saw with the non-looped cases)
// Correction: with the 3000-step runs the replay DOES decay, to 0.03-1 of its initial value (Fig. F2). In this particular panel the replay looks flat because it is normalised at t=1 and the decay is slow on a log axis; the 3000-step curves end at 0.05-0.3 for most models.
(saturation shouldn't be a problem because in trained LLMs it's reasonable that there is a maximum distance bc normalization layers bounds the norm of the hidden states) (or it might indicate that the LLM feedback achieved like a fixed point or some cycle, idk, i didn't look into it)
// Your second hypothesis is the right one, and it is now measured (Fig. F4): 7 of 11 models end in an exact token cycle (period 1-3; Qwen2.5-3B repeats " propulsion" forever, Mamba does a 3-cycle "\nA\n", Phi-4 a 2-cycle), and the others repeat a paragraph or keep writing coherent text (Qwen3-4B). The gain freezes once the orbit is self-consistent, so the saturation is a property of the attractor, not only of the norm bound.
We clear differences between trained LLMs and a random control. random control amplifies *much* less, and also it has a different shape (I can fit almost perfect)
// Right: the random control's gain is 3.8 nats vs 5-11 for trained models, and its fit is essentially perfect (R2=1.00) with tau=1480 steps, i.e. much longer than the run. Over 3000 steps that is indistinguishable from a pure exponential, which is expected: its margin never exceeds ~0.1 nats, so its gate never closes and it never self-regulates.

![Fig. F2b](update_figures/pack_loop_feedback.png) (b)
We clearly see that the randQwen random control stays constant at 1, while the rest are all positive. Most are either constant (with noise) or increase, but all of them are constant in the end.
// Panel (b) plots a_c/a_r, i.e. exp(L), so "constant at 1" = L=0 = no amplification. Panel (a) differentiates it, which is why it is noisy; the robust summary is the saturating fit L(t)=L_inf(1-exp(-t/tau)) with no offset (L(1)=0 by construction), reported in the table in section F.

![Fig. F9](update_figures/pack_sampling.png)
We compare the closed loop with other sampling methods to control for this, and ensure that the divergence is caused by deterministic feedback, not by sampling (still trying to understand this). the result is positive.
// Sharpen: CRN feeds a *hard token* sampled from the reference distribution, the same token to every row; independent feeds a hard token per row. Both replace the expected embedding by a one-hot, so they remove the mean feedback as well as the sampling noise. The measured result is much stronger than "not caused by sampling": soft (mean) grows 10^1-10^4 while CRN stays at 0.03-1.3 and independent at 0.8-6 over 3000 steps (Fig. F5/F5b). So the instability requires feeding back the *distribution*, not a sampled token.

![Fig. F13](update_figures/pack_loop_metric_T.png)
![Fig. F14](update_figures/pack_loop_fsle.png)
![Fig. F17](update_figures/pack_loop_finitetime.png)
all try to quantify the growth rate. from previous figures.
// Keep, but state what each is: M is the mean log amplification over the run (Fig. F7); FSLE is the first-passage time to a threshold S, i.e. log S / t_S (S = factor by which the response has grown relative to t=1); the finite-time exponent is (1/tau) log(a_{t+tau}/a_t) and it goes to ~0 once the response saturates - that is consistent with the flat ratio in F2b, not contradictory. FSLE needs at least two thresholds reached to be meaningful; models whose response never reaches S=8 give a single point.

![Fig. F4](update_figures/pack_gate_law_loglog.png)
![Fig. F5](update_figures/pack_gate_law_per_model.png)
Shows that each softmax is dominated by the top 2 tokens but this breaks down at very large temperatures. This shows how many tokens start being relevant as a function of temperature T.
I need to improve this explanation.
// Improved version. What is plotted is the largest singular value of the two-token readout covariance, sigma_max(Sigma_p), against m/T, where m is the top1-top2 logit gap and T the temperature. For a two-token distribution the identity is sigma_max = (G/4) sech^2(m/2T), i.e. slope -1 in log-log for m/T > 1, where G is a per-model pair-geometry constant. So:
//  * the -1 slope is NOT a discovery, it is the softmax identity; what we measure is whether the data follow it;
//  * trained models and the random control all show slope -1 to within 0.1 when fitted for m/T>1 (randQwen -1.01, R2=0.99) - so "the identity breaks down" is wrong; the earlier claim came from fitting through the flat m/T<1 region;
//  * what does NOT match is the *amplitude*: the data sit below the identity curve for most models, which means G_eff is smaller than the two-token prediction, i.e. the readout is not exactly two-token;
//  * "how many tokens are relevant" is exactly what the gate law measures: when m/T < 1 several tokens contribute, when m/T >> 1 the distribution is effectively one-hot.

![Fig. F19](update_figures/pack_loop_selfreg.png)
![Fig. F19b](update_figures/pack_loop_selfreg_big.png)
are quite brilliant at explaining the saturation: the LLMs reach a repeated loop. either like a stable point or fixed cycle. they get stuck there. more evidence and figures needed to communicate this better.
// Now have the direct evidence: Fig. F4 shows the period of the reference orbit for every model (period 1 = fixed point, 2-3 = short cycle, "none" = a repeated paragraph or continuing text) and decodes what is repeated. Note the green shading was removed as you asked, and the "one-shot ratchet" jargon is now "the loop switches to a different regime and stays there".



![Fig. F2](update_figures/pack_loop_metric_M.png)
you clearly see how there is amplifcation only in the closed loop case.
// Right, and with 3000 steps: M(closed) = 1-7 nats, M(replay) and M(CRN) at or below 0, M(independent) ~0.

![Fig. B4b](update_figures/pack_depth_tailfit.png)
the cumulative gain increases as the token goes through the layers
// Careful: cum_l is the linearised gain of the *optimal* direction, and it is dominated by block 0 (50-90% of the total log gain). It is a statement about the Jacobian product, not about how significant a given perturbation becomes. For the latter use the relative response R_l = ||dh_l||/||h_l|| (Fig. B5/B0b).

![Fig. B5](update_figures/pack_depth_relative.png)
The relative response initially increases. for small models it either keeps increasing or stays constant and larger than 1. for large models, it decreases below 1. Control missing. Not very conclusive.
// Now conclusive, and the controls are added (Fig. B5b, hatched panels). Numbers at t=50: R_L/R_1 = 43 (Phi-4-mini), 4.0 (Qwen2.5), 2.3 (DeepSeek-1.5B), 1.5 (Qwen3-4B, Qwen3.5-2B), 1.55 (randQwen), 1.17 (Mamba), 0.61 (Pythia), and 0.33-0.43 for the three 14B models. So the answer to "do perturbations become more significant through the layers" is NO: they are conserved or decay, and the absolute growth (8-1880x) is almost entirely the residual-stream norm growth (2.5-266x). The exception is Phi-4-mini, which really amplifies relative to its state.

![Fig. F5b](update_figures/pack_gate_linlog_per_model.png)
See the fit for every model
// Reading: red = two-token identity with the fitted G; black dashed = tail slope. The identity is above the data at large m/T for most models; Mamba and Qwen3-4B are closest. The robust quantity is the tail slope (-1), not the amplitude.

![Fig. D1](update_figures/pack_time_all_models.png)
clearly understand how teacher forced generation removes the "discrete sampling"
![Fig. D3b](update_figures/pack_time_layers_controls.png)
- what is the W tho?
// W = 100 for every A1 run behind these two figures (300 generated tokens, r=0.1, 512 directions). The 14B models use no window (W=0/4096). This matters: with W=100 the response leaves each layer at t*_l = W-P+p_last+(l-1)(W-1), which is the ~100-token staircase you noticed in D3c - it is the window law, not a model property (Fig. E5). The loop figures (F2 onward) use W=4096, where no such staircase exists inside 3000 tokens.
