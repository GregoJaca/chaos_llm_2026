

## Analytical Framework for Chaotic Dynamics in Transformer Architectures

To formalize the empirical observations of trajectory divergence and fractal dimensionality, we construct an analytical framework treating the Transformer architecture as a discrete delay-coordinate dynamical system. The emergence of deterministic chaos—characterized by exponential sensitivity to initial conditions within a bounded attractor—necessitates two competing topological forces: continuous phase-space stretching and nonlinear folding. In modern large language models, these forces map directly onto the specific architectural subcomponents of the Transformer block.

### 1. Nonlinear Stretching via Self-Attention

The dot-product self-attention mechanism is a source of phase-space expansion in Transformers. We analyze the propagation of an infinitesimal perturbation $\delta X$ through the attention operator:


$$Y = A X W_V $$
$$Y = \text{softmax}(d_k^{-1/2} X W_Q W_K^T X^T) X W_V $$


where the attention matrix $A$ is the row-wise softmax of the scaled logits $A = \text{softmax}(Z)$ and $Z = d_k^{-1/2} X W_Q W_K^T X^T$.

$$ A_{i,j} = \frac{\exp(x_i W_{QK} x_j^T)}{\sum_{k} \exp(x_i W_{QK} x_k^T)} $$
% GGG


Taking the total matrix differential $dY$ with respect to $dX$ yields:


$$ \delta Y = (\delta A) X W_V + A (\delta X) W_V $$


The critical source of instability lies in the differential of the pre-softmax logits:


$$ \delta Z = \frac{1}{\sqrt{d_k}} \left[ (\delta X) W_Q W_K^T X^T + X W_Q W_K^T (\delta X)^T \right] $$

Because the input matrix $X$ appears quadratically in the exponent of the softmax function, the gradient of the attention weights with respect to $X$ contains terms proportional to the input values themselves. 

% GGG ---------
$$ \delta A = A \odot \left( \delta Z - (A \odot \delta Z)\mathbf{1}\mathbf{1}^T \right) $$

Because $\delta Z$ is a linear function of $X$ (as seen in the first equation), substituting $\delta Z$ into the second equation mathematically proves that the Jacobian $J_{Attn} = \frac{\partial A}{\partial X}$ is directly proportional to the values of $X$. Since $A$ contains exponential terms ($e^Z$), the derivative incorporates the product of exponentials and $X$.
% GGG ---------


As proven by Kim et al. (2021), this quadratic dependency ensures that standard dot-product self-attention is not Lipschitz continuous for unbounded domains (GGG is it relevant? are our domains unbounded lol?). The spectral norm of the attention Jacobian, $\|J_{Attn}\|_2$, is therefore unbounded. 
% GGG
When operating in diffuse attention regimes (the linear region of the softmax), the quadratic coupling acts as a local amplifier, meaning a small change in the input ($\delta X$) causes exponentially large changes in the attention weights. 
% GGG 

### 2. Anisotropic Linear Stretching in MLPs % GGG this whole part is super weak claim, we just bound by above, and say that the bound is high.lol

The Multi-Layer Perceptron (MLP) block applies token-wise nonlinear transformations that further stretch the phase space. For a single token, the MLP computes $f(x) = W_2 \phi(W_1 x)$. 

$h = W_1 x$.

The differential is:$$ \delta h = W_1 \delta x $$

Let the post-activation vector be $a = \phi(h)$. Because the activation function $\phi$ is applied element-wise to the vector $h$, a perturbation $\delta h$ scales by the derivative of $\phi$ at each element:
$$ \delta a = \phi'(h) \odot \delta h = \text{diag}(\phi'(h)) \, \delta h $$

Let the final output be $y = W_2 a$. 
$$ \delta y = W_2 \delta a $$
$$ \delta y = W_2 \left[ \text{diag}(\phi'(W_1 x)) \, (W_1 \delta x) \right] $$
$$ \delta y = \left[ W_2 \, \text{diag}(\phi'(W_1 x)) \, W_1 \right] \delta x $$

The local amplification of a perturbation $\delta x$ is governed by the Jacobian:


$$ J_{MLP} = W_2 \, \text{diag}(\phi'(W_1 x)) \, W_1 $$

The maximal stretching applied to $\delta x$ is bounded by the sub-multiplicative property of the spectral norm:


$$ \|\delta f(x)\|_2 \le \|W_2\|_2 \, \|\phi'\|_{\infty} \, \|W_1\|_2 \, \|\delta x\|_2 $$


Standard LLM pre-training does not apply spectral normalization to constrain the operator norms of $W_1$ and $W_2$. Consequently, gradient descent optimization naturally drives the dominant singular values of these matrices well above unity to maximize feature expressivity. As demonstrated by Cowsik et al. (2024), who observed order-to-chaos phase transitions in Transformers based on weight variance, these unconstrained weight matrices act as anisotropic stretchers, linearly amplifying perturbations along their principal right singular vectors.



### 3. Topological Folding via Normalization

To prevent numerical overflow instabilities Transformers use normalization layers, most commonly LayerNorm or RMSNorm. From a dynamical systems perspective, these operators strictly confine the $D$-dimensional hidden states, forcing trajectories to evolve on the surface of a bounded, lower-dimensional manifold.

For an input vector $x \in \mathbb{R}^D$, Root Mean Square Normalization (RMSNorm) computes $y = \frac{x}{\text{RMS}(x)}$, where $\text{RMS}(x) = \frac{1}{\sqrt{D}} \|x\|_2$. The Jacobian of this transformation with respect to a perturbation $\delta x$ is:

$$ J_{RMS} = \frac{\sqrt{D}}{\|x\|_2} \left( I - \frac{x x^T}{\|x\|_2^2} \right) $$

Standard LayerNorm applies the same operation to the mean-centered vector $\tilde{x} = x - \mu \mathbf{1}$, scaled by the standard deviation (GGG or variance?) $\sigma = \frac{1}{\sqrt{D}} \|\tilde{x}\|_2$. Its Jacobian is:

$$ J_{LN} = \frac{1}{\sigma} \left[ \left( I - \frac{1}{D}\mathbf{1}\mathbf{1}^T \right) - \frac{\tilde{x} \tilde{x}^T}{\|\tilde{x}\|_2^2} \right] $$

Both Jacobians share the structure: $J_{Norm} = \frac{1}{S} P_{\perp}$, where $S$ is a scaling factor proportional to the variance or magnitude of the state, and $P_{\perp}$ is an orthogonal projection matrix. This structure confines chaotic divergence through two distinct mechanisms:

**Radial Damping**
As the Attention and MLP layers stretch the vector $x$ along their expansive directions, the absolute magnitude ($\|x\|_2$) or variance ($\sigma$) increase. The normalization Jacobian inversely scales any incoming perturbation $\delta x$. 
% (GGG hard to claim it contracts every perturbation, bc how does it know what a perturbation is?, like if a perturbation reduces the variance of a vector, wouldn't normalization then expand the vector? but maybe here the expansion goes against the perturbation? maybe the claim is not that norm contracts all vectors, but that it contracts all perturbations)

**Orthogonal Annihilation**
The matrix $P_{\perp}$ dictates the geometry of the permitted dynamics. In RMSNorm, $P_{\perp} = \left( I - \frac{x x^T}{\|x\|_2^2} \right)$ is an exact orthogonal projection onto the tangent space of a $D$-dimensional hypersphere. It satisfies $P_{\perp} x = 0$. In LayerNorm, the equivalent projection satisfies $P_{\perp} \tilde{x} = 0$. Mathematically, this means any component of an expansive perturbation $\delta x$ that points outward in the radial direction of the activation vector is strictly annihilated.

Together, these mechanisms dictate that expansive forces cannot push trajectories to infinity. Instead, trajectories are compressed onto the spherical manifold, forcing the divergence to manifest as angular sliding along the surface. This satisfies the requirements for a strange attractor, justifying the low-dimensional, fractal correlation dimensions observed in the deeper layers of the model. % GGG meh

### 4. Spectrum Shifting via Residual Connections

LLMs use residual connections: for any sub-block $F(x)$, the output is $x_{res} = x + F(x)$, yielding the total Jacobian:

$$ J_{res} = I + J_F $$

The eigenvalues  $\lambda_i$ of $J_F$ are shifted to $1 + \lambda_i$, so for any positive eigenvalue of the Attention, Normalization or MLP blocks, residual stream's spectral radius is strictly larger than 1.


### 5. Latent Transmission and the Delay-Coordinate Jump 
% GGG rewrite style (and clarify that this part of out "macroscopic" experiments)

LLMs have a continuous 

The interaction between the continuous latent stretching and the discrete step-function divergence observed in generated text is reconciled by treating autoregressive inference as a delay-coordinate system. The hidden state at step $t$ is explicitly coupled to the history of all past tokens via the Key-Value (KV) cache:


$$ h_t = \sum_{j=0}^{t} A_{t,j} V_j $$

An initial prompt perturbation $\delta X_0$ is permanently encoded in the cache as $\delta V_0$. Consequently, the perturbation at step $t$ is:


$$ \delta h_t = \sum_{j=0}^{t} \left[ (\delta A_{t,j}) V_j + A_{t,j} (\delta V_j) \right] $$

This discrete delay-differential equation reveals that the historical perturbation is continuously re-injected into the present state. The latent error $\delta h_t$ grows exponentially through the depth-wise Attention and MLP Jacobians at every generation step. However, greedy decoding applies a discrete filtering function $y = \text{argmax}(W_U h_t)$. The generated text remains entirely static until the latent perturbation strictly exceeds the logit confidence margin:


$$ (\delta z_k - \delta z_y) > (z_y - z_k) $$


The apparent step-function divergence in macroscopic text distance metrics is therefore not an artifact, but the precise first-passage time at which the continuous, chaotic latent growth breaches the discrete categorization threshold.













Normalization is a local folding mechanism. It projects the vector back to the sphere at layer $\ell$, but because of the residual connection $x_{\ell+1} = x_\ell + F(x_\ell)$, the Jacobian of the composition of layers can still exhibit a positive Lyapunov exponent. We must clearly distinguish between local vector bounding (which LayerNorm achieves) and global trajectory stability (which LayerNorm fails to achieve, allowing chaos to persist).














% -------------

% G: (for a layer)
Let the phase space be the hidden state representation. We consider the propagation of an infinitesimal perturbation $\delta x$ through a layer map $f(x)$. The relation is given by the differential $df(x)$ evaluated at $dx = \delta x$.

* For vector-to-vector maps, this is the Jacobian matrix $J = \frac{\partial f}{\partial x}$, where $\delta x_{out} = J \delta x_{in}$.
* The **spectral norm** (induced 2-norm) of a matrix $A$ is $\|A\|_2 = \sigma_{max}(A)$, which is the largest singular value. It defines the maximum possible stretching of a vector: $\|Ax\|_2 \le \|A\|_2 \|x\|_2$.
* If $\|J\|_2 > 1$, the operator locally stretches perturbations along the direction of the dominant right singular vector.




% -------------






In chaos theory, "stretching" refers strictly to the amplification of a perturbation magnitude $\|\delta x\|$, not the absolute magnitude of the state vector $\|x\|$.If a system evolves via a map $x_{n+1} = F(x_n)$, the evolution of an infinitesimal perturbation $\delta x$ is governed by the Jacobian matrix $J = \frac{\partial F(x)}{\partial x}$:$$ \delta x_{n+1} = J(x_n) \delta x_n $$A component "stretches" the phase space if the spectral norm (the largest singular value) of its Jacobian exceeds 1 ($\|J\|_2 > 1$). When the product of Jacobians across layers and time steps maintains a spectral radius $> 1$, the system exhibits exponential sensitivity to initial conditions (a positive maximum Lyapunov exponent).



"the trajectories remain close, and then suddenly jump at a specific token. This happens because the system couples a continuous phase space (hidden states) with a discrete sampling bottleneck (the vocabulary)."


"The output token remains the same for some time, but the perturbation propagates through the attention mechanism. Autoregressive attention is not Markovian; it is a delay-coordinate system."


"At generation step $t$, the attention output is:$$ O_t = \sum_{j=0}^{t} \text{Softmax}(Q_t K_j^T) V_j $$Even if the current query $Q_t$ is unperturbed, the keys and values from the initial prompt ($K_0, V_0$) physically contain the initial perturbation $\delta K_0, \delta V_0$. The attention mechanism computes a weighted sum over the entire past.Therefore, the perturbation does not need to survive through the discrete output text. It is permanently cached in the continuous latent space, repeatedly stretched by the non-Lipschitz attention matrix, and folded by LayerNorm at every single autoregressive step, until the accumulated perturbation $\|\delta z\|$ at the final layer exceeds the logit margin, causing the sudden step-function jump in the text output."

Lipschitz constant $L$ is the supremum (the absolute global maximum) of the spectral norm of the Jacobian across the entire domain:$$ L = \sup_x \|J(x)\|_2 $$




------------------
power law perturbation








---

### How SwiGLU Affects the Dynamics (Compared to standard ReLU/GeLU)

From a dynamical systems perspective, SwiGLU is qualitatively different from older activation functions.
* **ReLU/GeLU (Univariate Activation):** $f(x) = W_2 \phi(W_1 x)$. The Jacobian is $\mathbf{J} = W_2 \text{diag}(\phi') W_1$. The input $x$ only enters the Jacobian through the derivative of the activation function $\phi'$. Since $\|\phi'\|_\infty \le 1$, the input cannot scale the Jacobian beyond the norm of the static weights.
* **SwiGLU (Multiplicative Coupling):** $f(x) = W_{\text{down}} (\text{SiLU}(W_{\text{gate}} x) \odot W_{\text{up}} x)$. Differentiating SwiGLU yields:
  $$\mathbf{J}_{\text{gate}} = W_{\text{down}} \text{diag}\left((W_{\text{up}} x) \odot \text{SiLU}'(W_{\text{gate}} x)\right) W_{\text{gate}}$$
  Notice that the pre-activation state vector $h_{up} = W_{up} x$ is **inside the diagonal scaling matrix**. 
* **The Consequence:** The Jacobian scales **linearly with the norm of the input state $\mathbf{x}$**. If the hidden state magnitude increases, the stretching factor (stiffness of the map) increases proportionally. 
* **Physical Analogy:** This is a **parametric amplifier** or **multiplicative feedback loop**. (In ML research, this quadratic scaling is known to cause "activation spikes" and numerical instabilities in FP8/FP16 at scale). SwiGLU models naturally support a positive feedback loop: larger states lead to larger Jacobians, which lead to even larger states, driving stronger chaotic stretching.

---