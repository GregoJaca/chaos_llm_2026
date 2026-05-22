put together everything I have for the thesis
rant what I know
say I want to say
write the sentences
send to János
think what I'm embarrassed to show
fix it


Write down what I don't understand. Be humble. Ask Janos. Speak, then listen. Listen.
Do this with:
-content
-background knowledge (math)
-communication style / strategy 
-actual writing 

Pdf is the proof.

Work on the things that work

What sucks?
What matters? What is important?
Write down together what is missing.
Do it.

Finish compute by Monday. Then run aggressively.

What is success criteria? What is nice to have? What are red flag?

What can Janos tell me that AI can't?

What do I want to say?
What will they understand?
What does a great thesis look like? How can I make mine be like it?















what I don't know:













story:
we want to study LLMs with the tools of dynamical systems an chaos theory. why? it's fucking interesting and no one is doing it as well as me. it lends itself nicely to study numerically, and the system itself is the model. in physics, we often sacrifice a lot of detail and nuance when we do a model of the system (which, especially in chaos and non-linear dynamics, can be bad bad). but in LLMs, the LLMs is already the model and you can directly do inference on it.
also it can hopefully give us insights into how these models operate.

throughout this paper we will look at two types of trajectories: 
- trajectory of a single token across the layers of the model. Or it could be a lot of tokens, but across the layers of the model. Here "time" is the layer index. problems: there are few layers (around 30). there is no clear directionality.
- trajectory of the sequence of tokens/hidden states. Look either at the token, or at the hidden state at a given layer. then look at the sequence that is generated autoregressively. this is nice because it allows for long trajectories. also the autoregressive generation is like a time evolution in the sense that there is a directionality or ordering. time here is the token index.

there are challenges associated to the discreteness of time and space. 
sampling (for autoregressive generation) is discrete and you select a single token, then that gets converted to hidden state (using the vocabulary matrix)

our hypothesis is that chaotic behavior is strongly dependent on attention. So we perform analyze some chaotic metrics as a function of the attention window size.
We expect the relationship not to be monotone or simple. 
We imagine attention to have many effects simultaneously (this is speculation): 
1. If context winow size = 0, then the LLM is just a feed forward nn. perturbations to a previous tokn cannot propagate. also there would be no long term coherence. in practice, the LLM just spits out nonsense, and then stuck in a loop (this is true).
2. At the same time, attention is responsible for propagating information. 
- this can be stabilizing in some cases. I imagine: let's say the whole context between two trajectories is identical, but the last token has a small perturbation. Then, the context could dominate over the last token and stabilize the trajectory. 
- it can cause divergence: a perturbation propagates and gets amplified (this is to be measured empirically) by attention and MLP, and then eventually causes divergence. I have observed this. proof: I applied perturbation to prompt. then inference. the first n tokens were the same (but perturbation)

I really fucking want to show how "how chaotic a system is" depends on the attention. my problem: I don't have a good scalar measure for "how chaotic a system is". I can't measure the lyapunov exponents. what I did is look at the divergence by measuring the distance in hidden state space. I used a lot of different distance metrics: cosine similary, hausdorf distance, dtw, frechet, correlations, etc... the result was a step-function and you had a jump in the distance when there was a flip in the selected token. the distance chage was huge (step function) and we had no continuously evolving distance. it was a sudden change. I tried to improve it by using sliding window, and other things, but it was impossible. So then I said: hey there is a step function behavior and the only relevant thing I can measure is how much time it takes for the system to diverge (and to get this first different token). for this I don't need to look at hidden states, I can do this analysis directly with the token (as integers or strings). It is unfortunate that a lot information is lost, but I hope this still will be useful. This is simpler to look at as a function of the attention context window size.

we definitely see a dependence on the attention, but it is not straight forward and it depends on the magnitude of the perturbation. so it is really hard for me to report my results, since they are not very clear and they depend on a lot of things. Also I did it for a single prompt, and that also makes it hard to compare. it is hard to find a single parameter dependence, and I don't know how to present my results.

at the same time, we seek a more "mechanistic" explanation for the behavior. we note that in most chaotic systems, which form a strange attractor for example, there is stretching (positive lyapunov exponents, divergence) and folding (keeps the trajectories bounded in the strange attractor) at the same time. And these two phenomena together make this a chaotic system.
that's why we looked also at the trajectory of a single token across all layers (and creating a lot of slightly perturbed initial conditions)

my hope is to be able to put everything together. I will tell you the analysis from most macroscopic to most microscopic


- recurrence plots of single trajectories, and finding really interesting plots, with similarities to recurrence plots from chaotic systems. especially when doing this with the hidden states of the last layer of the system
- measuring the correlation dimension of a single trajectory. When this is done with the hidden states at the first layer of the system, this looks like a random set of vectors, but when done with the hidden states from the last layer of the model, we see exactly what we expect for a deterministic chaotic system.
- measures the index of token divergence (as a function of attention window size mainly, but also perturbation magnitude). 
- look at how a perturbation propagates across layers. this is the run_perturbations analysis. This doesn't distinguish between MLP, attention, norm, residual, activation function, etc... , but we see this nice power law behavior in how the magnitude of the perturbation at the middle layers (excluding the first and last 5) as a function of the magnitude of the original perturbation. This is common in a linearized system. 
-Then we also look at the exact jacobian of MLP

in many different scales and and across many different analysis, we see properties similar to those of chaotic systems. we look for the explanations of this behavior in the architectural properties and design of LLMs, then we also measure some of these (like jacobians)


TODO:
see what we can measure easily from attention matrices
do more runs different prompt, 

write simplex algo. use algorithm style




QUESTIONS:
measure power law exponent, why is it almost 1 or 10? maybe the 10 comes from the costant shift
why is there a constant "background noise"? can we explain the magnitude of it?
G: maybe bc of linear + activation function which might ?
go to 0
