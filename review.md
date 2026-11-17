Reviewer Comments:
Reviewer #1 Evaluations:
Overall Rating: Reconsider based on responses to issues raised by reviewers

Reviewer #1 (Comments to the Author):

The authors apply tools from nonlinear dynamics to study and characterize the hidden-state trajectories of an LLM. The authors perturb input embeddings by a fixed-norm random vector, evolve the perturbed and unperturbed trajectories through the network, and track divergence with multiple distance metrics to characterize the dynamics, thus demonstrating similarities with chaotic systems such as the Lorenz attractor. This is an interesting study addressing a topic of growing importance, and I consider it potentially suitable for publication following major revision. My principal concerns relate to the interpretation of the results as evidence of deterministic chaos, which is central to the manuscript.

1. Jacobian norms exceeding unity for individual components demonstrate possible local amplification, but do not establish a positive Lyapunov exponent for the complete evolution. The authors should clarify how their component-level measurements support conclusions about the full dynamics.
2. Positive finite-time expansion can also occur in nonchaotic systems and therefore does not, by itself, establish chaos in the dynamical-systems sense. The authors should distinguish transient perturbation amplification and divergence induced by discrete token-selection thresholds from evidence of deterministic chaos.
3. The values of the recurrence-rate parameters (fixed at 0.03 or 0.05) and distance threshold choices should be justified or shown to be robust to reasonable variation.
4. The manuscript should report convergence diagnostics for the correlation-dimension estimates, particularly their dependence on the number of sampled points.
5. The conclusions should be restricted more explicitly to the models used and experimental settings examined. Broader claims about chaotic dynamics in LLMs therefore require validation across additional models, architectures, and scales.
6. The manuscript should clarify what additional explanatory or predictive value follows from interpreting the observed behavior as chaos.


Reviewer #2 Evaluations:
Overall Rating: Reconsider based on responses to issues raised by reviewers

Reviewer #2 (Comments to the Author):

This paper studies whether large language models (LLMs) show chaotic behavior during text generation. The authors examine how small changes in initial embeddings affect generated outputs. They also analyze the Jacobians of different Transformer components and study the structure of hidden-state trajectories. The topic is extremely interesting and of real importance, and the analysis of individual Transformer components is potentially useful. With that said, I have several major concerns.

- Does the paper really provide evidence of chaos?

The main result is that two trajectories with slightly different initial conditions can remain similar for many steps and then suddenly diverge. However, this behavior does not necessarily imply chaos. In greedy decoding, a small change can cause the model to select a different token, then the subsequent outputs may become very different, even if the underlying system is not chaotic. The authors need to distinguish this vs genuine chaotic behavior.

- Does the Jacobian analysis actually support the main claim?

The authors show that the Jacobians of attention and feed-forward layers often have spectral norms greater than one. However, this only shows that these components can amplify small changes locally. It does not establish that perturbations grow exponentially over time. The combined effect of different components may, possibly, even reduce perturbations.

- Are the other results specific to chaos?

The recurrence plots and correlation-dimension analysis show interesting structures in the hidden-state trajectories. However, similar patterns can also arise in non-chaotic systems with strong temporal or semantic correlations. These results are suggestive, but they do not clearly distinguish chaotic dynamics from other forms of structured behavior.

- What is the main new contribution?

This is my broader concern about the paper. Several findings may be explained by properties of Transformers that are already well understood. Transformers are nonlinear systems, their components can amplify perturbations, and small changes in token selection can lead to very different outputs. The authors need to explain more clearly what new scientific insight their analysis provides beyond describing these known properties using the language of chaos theory.

The connection to chain-of-thought reasoning is also somewhat unclear to me. Most of the mechanisms studied here apply to ordinary autoregressive text generation. The paper does not show that chaotic behavior is specific to reasoning or that it affects reasoning performance.

Overall assessment

I find the research question absolutely fascinating, and some of the technical analysis are indeed valuable. However, the main claim that LLMs exhibit deterministic chaos is not sufficiently supported by the evidence. More importantly, it is not yet clear which findings represent new insights rather than alternative descriptions of known Transformer behavior. I therefore cannot recommend publication in its current form. A substantially revised paper with clearer claims, suitable non-chaotic comparisons, and a stronger explanation of its contribution could be valuable.