the bad part: we got many negative results and have to change many claims from the paper
the good part: i believe the results we got, and the understanding we get is much more interested and nuanced. also reviewers asked for "major revisions"; we have that.
the challenge: how can we most effectively communicate all of this and make a coherent story and explanation.


My current idea of how to frame the paper would be:

we study LLMs as dynamical systems. we studied the sensitivity to initial conditions and perturbations in a deterministic (greedy sampling T=0).

1 we studied autoregressive generation and compared unperturbed trajectories with perturbed ones. one of the findings was that for a big enough perturbation the trajectories did not immediately diverge, but rather they stayed relatively close for some time - and then eventually once they sample a differet token - they diverge quickly.
we could not really explain if the divergence was coming from the model itself or only from the sampling. (reviewers also marked this)

2 we studied how a single token goes through the transformer layers. looked at jacobians, how much perturbations are expanded at each operation (MLP, attention, normalization, etc...). some of the work is analytical, some is experiments, both with exact jacobiand and with finite size perturbations.
we found a power law relationship - which breaks down at very karge perturbations, and has a noise floor at very small perturbations - showing that that the jacobian analysis (linearity) is valid.
what we show is that the layers have the *potential* to amplify perturbations.
(reviewers mention that this doesn't confirm it, nor confirms this as the cause for divergence, which is true)

now the new part

3 to address the limitations of (1) we use teacher forcing.
teacher forcing: you do autoregressive generation for 1 unperturbed trajectory (teacher). then for the perturbed trajectories you force them to sample the same token as the teacher, but the hidden states have the perturbation. this eliminates the "divergence caused by sampling" and isolates the effect of the model itself.
key results:
- the LLMs are not expanding the perturbations
- perturbations are propagated forward by attention, and the perturbations at layer

also something worth mentioning, it kinda makes sense that the model is behaving in this way
An analogy: imagine the perturbed initial state as water with high salt concentration. At each step a new token is added, and it’s not perturbed (although the previous perturbed tokens will perturb its hidden states), so it’s like adding clean water to the mix. Eventually the salt concentration will decrease. The attention window size is a key parameter that regulates for how long the original perturbation remains)


additional results (Question: is this downstream of (3) or does it have a different experimental setup?):
- also we found tha t "selecting another token in the autoregressive case" is not correlated with the perturbation amplification. rather it depends almost exclusively on "logit margin" = "how far apart are the top1 and top2 predictions". it seems like it doesn't depend on the size of the perturbation (for small enough perturbations)
(Question here: are we sure? does it depend *almost exclusively* on logit margin)
(from experience if the perturbation is huge, the first generated token will be some chinese bs, but at least the results seem to indicate that for the small perturbation (linear) regime, the main predictor of when the first sampled token will be different is the "logit margin", not size of perturbation)

- because we are in the linear regime, it's possible to calculate "for a given perturbation direction, what is the minimum perturbation magnitude that changes a token" (Question: here it's the perturbation of the prompt or of the last layer hidden state that produces the logits?)


Questions: 

4 we expand on (2) and do the analysis not for individual layers, but for the composition of the layers. so the real deal

key results:
- while each individual layer amplifies perturbation in certain directions (singular vectors), these directions tend to be rotates 87° from layer to layer. this makes perturbations not dominate, and keeps stability.
- we see that perturbations grow in the first layer(s) and then stays almost constant (for most LLM models)
- tangent contraction rate decreases wrt layers.
- the hidden state norm grows a lot. the perturbation norm grows (mainly in te first layer), but way less. so the relative perturbation norm decreases. 
- in Qwen3.5-9B we see perturbation norm growing significantly in every layer that has the Full Attention block, and to very slightly decrease in layer with Gated DealtaNet attention. interesting!

Questions: 
how does this look like in controls? 
tangent contraction rate is not very clear or well defined to me mathenatically. if i see the code maybe it's more clear. is there a simpler term? like saying: "the contraction rate decreased" isn't it the same as "the expansion rate increased" which i guess is simpler? can we use more consistent and clear language?

5 We explore runs with non-zero Temperature T

Questions: do we only use T =/= 0 in teacher forced case?

6 Linear response and Kernel

for most experiments, to generate a perturbed intial condition, we take the whole prompt (~100 tokens), tokenize it and get the hidden states in the first layer (just with the vocab matrix) and add a perturbation tensor to all of it. dim(tensor) = n_prompt_tokens x hidden_dimension

the research question here is: can we approximate this with a kernel? is the effect of perturbing the full prompt the same as the sum of the effect of perturbing each token in the prompt individually?
like a Green's function :)

key results:
- it seems like yeah but i need to check this more in depth. idk if this should go in this paper, maybe appendix? idk if it's immediately relevant. 
- this seems to hold for a narrow range of "perturbation magnitude". for very small perturbations, noise floor dominates, for large perturbations, it saturates.


7 Flux tubes and stable chaos
dynmics are stable to small perturbations, but large enough perturbations (above a critical size "flux tube radius" , remember the "for a given perturbation direction, what is the minimum perturbation magnitude that changes a token"?) there is a discrete event which causes trajectories to diverge. in our case, the "flux tube radius" is prompt, trajectory and model dependent (and also depends on T, attention and a million things). 
Here if you think that pedagogically it's a useful concept, we can mention it, but I'm not convinced (I don't want to overload the paper with too many new concepts, tho we can mention it or reference some papers). Otherwise we can just show the results and explain it in clear words.

(I'm still trying to make a good figure from this)

8 Controls:
We used several controls for most parts of our analysis, and we used many different controls. Some show negative results, e.g.: with mamba (a probably contractive transformer) we show that the LLMs we study don't amplify perturbations. Others show positive results:

9 Loops:


There are other interesting results we get
- we look at Qwen3.5-9B (which has two different attention types: full attention and gated DeltaNet) and we clearly see this in how perturbations propagate with it.


