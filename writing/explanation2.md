![Fig. C3](update_figures/pack_rotation_controls.png)

Top singular vectors of consecutive block Jacobians are close to orthogonal, but more aligned than random in trained LLMs. This explains why while each layer Jacobian has the *capacity* to amplify perturbations, the growth is limited.

![Fig. D3b](update_figures/pack_time_layers_controls.png)
clearly understand how teacher forced generation removes the "discrete sampling" (we don't have a "discrete" "step-function-like" increase) compared to what we have in autoregressive generation. 
In general we see that there might be a transient growth but eventually perturbations decay.
We see clear differences of the response at different layers.
In most cases, 
note: layer 1 is the *second* layer. layer 0 is always just a column of the vocabulary matrix, and thus constant (unless we perturb it)
This figure answers: Is the "discrete" "step-function-like" divergence you saw during autoregressive generation caused by the model itself or by the sampling (at each generation step). This answers: it was caused by sampling.
This is effectively a control that removes the sampling (because it samples the same as the teacher).

![Fig. F0d](update_figures/pack_loop_loglog_ds15.png)
Following up on the previous figure (update_figures/pack_time_layers_controls.png) what we do now is looping.
We compare the "closed loop" against the "network only replay" (control) and we see that the closed loop amplifies perturbations as it generates tokens. eventually it saturates. The "replay" stays roughly constant at ~1, which means that the perturbation doesn't get amplified. interestingly it doesn't decay (as we saw with the non-looped cases)
(saturation shouldn't be a problem because in trained LLMs it's reasonable that there is a maximum distance bc normalization layers bounds the norm of the hidden states) (or it might indicate that the LLM feedback achieved like a fixed point or some cycle, idk, i didn't look into it)
We clear differences between trained LLMs and a random control. random control amplifies *much* less, and also it has a different shape (I can fit almost perfect)

![Fig. F2b](update_figures/pack_loop_feedback.png) (b)
We clearly see that the randQwen random control stays constant at 1, while the rest are all positive. Most are either constant (with noise) or increase, but all of them are constant in the end.

![Fig. F9](update_figures/pack_sampling.png)
We compare the closed loop with other sampling methods to control for this, and ensure that the divergence is caused by deterministic feedback, not by sampling (still trying to understand this). the result is positive.

![Fig. F13](update_figures/pack_loop_metric_T.png)
![Fig. F14](update_figures/pack_loop_fsle.png)
![Fig. F17](update_figures/pack_loop_finitetime.png)
all try to quantify the growth rate. from previous figures.

![Fig. F4](update_figures/pack_gate_law_loglog.png)
![Fig. F5](update_figures/pack_gate_law_per_model.png)
Shows that each softmax is dominated by the top 2 tokens but this breaks down at very large temperatures. This shows how many tokens start being relevant as a function of temperature T.
I need to improve this explanation.

![Fig. F19](update_figures/pack_loop_selfreg.png)
![Fig. F19b](update_figures/pack_loop_selfreg_big.png)
are quite brilliant at explaining the saturation: the LLMs reach a repeated loop. either like a stable point or fixed cycle. they get stuck there. more evidence and figures needed to communicate this better.



![Fig. F2](update_figures/pack_loop_metric_M.png)
you clearly see how there is amplifcation only in the closed loop case.

![Fig. B4b](update_figures/pack_depth_tailfit.png)
the cumulative gain increases as the token goes through the layers

![Fig. B5](update_figures/pack_depth_relative.png)
The relative response initially increases. for small models it either keeps increasing or stays constant and larger than 1. for large models, it decreases below 1. Control missing. Not very conclusive.

![Fig. F5b](update_figures/pack_gate_linlog_per_model.png)
See the fit for every model 

![Fig. D1](update_figures/pack_time_all_models.png)
clearly understand how teacher forced generation removes the "discrete sampling"
![Fig. D3b](update_figures/pack_time_layers_controls.png)
- what is the W tho?
