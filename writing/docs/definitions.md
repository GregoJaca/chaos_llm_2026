

---

### 1. Self-Attention: Nonlinear Perturbation Stretching

We wish to determine how a perturbation $\delta X$ in the input sequence propagates through the attention mechanism.

Let $X \in \mathbb{R}^{N \times D}$ be a sequence of $N$ tokens. The dot-product attention operator is:


$$ Y = A X W_V $$


where $A = \text{softmax}(Z)$ applied row-wise, and the pre-softmax logits are:


$$ Z = \frac{1}{\sqrt{d_k}} X W_Q W_K^T X^T $$


where $W_Q, W_K \in \mathbb{R}^{D \times d_k}$ and $W_V \in \mathbb{R}^{D \times D_v}$.

To find the perturbation $\delta Y$, we take the total matrix differential $dY$ with respect to $dX$:


$$ \delta Y = (\delta A) X W_V + A (\delta X) W_V $$

First, we find the differential of the logits $\delta Z$. Applying the product rule:


$$ \delta Z = \frac{1}{\sqrt{d_k}} \left[ (\delta X) W_Q W_K^T X^T + X W_Q W_K^T (\delta X)^T \right] $$

Next, we evaluate the differential of the softmax matrix $A$. Let $a_i$ be the $i$-th row of $A$, and $z_i$ be the $i$-th row of $Z$. The Jacobian of the softmax function is $\text{diag}(a_i) - a_i^T a_i$. Thus, the row-wise differential is:


$$ \delta a_i = a_i \odot \delta z_i - a_i (\delta z_i \cdot a_i^T) $$

**Analysis of Stretching:**
The term $X$ appears multiplicatively with $\delta X$ in $\delta Z$. Previously, I stated this causes unbounded growth if $X$ is arbitrarily large. As you correctly pointed out, LayerNorm fixes the vector magnitudes such that $\|x_i\|_2 = \sqrt{D}$. Therefore, $X$ is bounded.

However, the magnitude of the entries in $Z$ scales with the dimension $D$. If the attention distribution $a_i$ is relatively uniform (far from a one-hot vector), the term $a_i \odot \delta z_i$ dictates that the perturbation is scaled by the local magnitude of the weights in $X W_Q W_K^T$.
Kim et al. (2021) prove that because $X$ appears twice in the exponent, the global Lipschitz constant is theoretically unbounded for the un-normalized domain. In the normalized domain induced by LayerNorm, the local spectral norm $\|J_{Attn}\|_2$ is governed by the projection of $\delta X$ onto the eigenspace of $W_Q W_K^T$. If the spectral norm $\|W_Q W_K^T\|_2$ is sufficiently large, the quadratic coupling $\delta X \dots X^T$ acts as a local amplifier, mapping small input perturbations into large variations in the attention probability mass $\delta A$.

*Honesty Note:* Self-attention does *not* always stretch. If $Z$ is highly peaked, the softmax saturates. In this regime, $\text{diag}(a_i) - a_i^T a_i \approx 0$, meaning $\delta A \approx 0$. Thus, attention acts dynamically as a router: it stretches perturbations heavily when attention is diffuse (operating in the linear region of softmax) and suppresses them when attention is highly confident (saturated).

---

### 2. The MLP Block: Linear Stretching

Let the Multi-Layer Perceptron block for a single token be $f(x) = W_2 \phi(W_1 x)$, where $x \in \mathbb{R}^D$, $W_1 \in \mathbb{R}^{H \times D}$, $W_2 \in \mathbb{R}^{D \times H}$, and $\phi$ is an element-wise activation function (e.g., SiLU).

By the chain rule, the Jacobian is:


$$ J_{MLP} = \frac{\partial f}{\partial x} = W_2 \, \text{diag}(\phi'(W_1 x)) \, W_1 $$

To find the maximal possible amplification of a perturbation $\delta x$, we take the spectral norm of the Jacobian. Using the sub-multiplicative property of matrix norms ($\|AB\|_2 \le \|A\|_2 \|B\|_2$):


$$ \|J_{MLP}\|_2 \le \|W_2\|_2 \, \|\text{diag}(\phi'(W_1 x))\|_2 \, \|W_1\|_2 $$

* $\|W_1\|_2$ and $\|W_2\|_2$ are the maximal singular values of the weight matrices.
* $\|\text{diag}(\phi'(W_1 x))\|_2 = \max_i |\phi'((W_1 x)_i)|$. Let this be bounded by a constant $c_\phi$ (for SiLU, $c_\phi \approx 1.1$).

Thus, the perturbation magnitude is bounded by:


$$ \|\delta f_{MLP}\|_2 \le c_\phi \|W_1\|_2 \|W_2\|_2 \|\delta x\|_2 $$

**Analysis of Stretching:**
We cannot mathematically prove that $c_\phi \|W_1\|_2 \|W_2\|_2 > 1$ as an axiom of linear algebra. This is strictly a property of the converged weights post-training. However, because LLM pre-training does not constrain the operator norm of $W_1$ and $W_2$ (via spectral normalization), gradient descent dynamically favors large singular values to increase feature separation. Consequently, along the principal right singular vectors of $W_1$, the MLP block acts as an anisotropic stretcher.

---

### 3. Skip Connections: Spectrum Shifting

You correctly identified that skip connections limit the structural deviation. Let $F(x)$ represent either the Attention or MLP sub-block. The residual architecture maps $x_{out} = x_{in} + F(x_{in})$.

The Jacobian of the residual block is:


$$ J_{res} = I + J_F $$

Let $\lambda_i$ be an eigenvalue of $J_F$ with eigenvector $v_i$. The corresponding eigenvalue for $J_{res}$ is exactly $1 + \lambda_i$.

**Analysis of Stretching:**
This is a critical mathematical feature for chaotic dynamics. If $J_F$ is strictly contractive ($\lambda_i < 0$), the residual stream maintains the eigenvalue near 1 (assuming $|\lambda_i| < 1$). Conversely, if $F(x)$ produces *any* expansive direction ($\lambda_i > 0$), the eigenvalue of the full block becomes $1 + \lambda_i > 1$. The identity matrix prevents the hidden state from collapsing to zero, shifting the spectrum such that expansive perturbations accumulate monotonically along the residual stream unless forcefully damped by normalization.

---

### 4. Normalization (LayerNorm/RMSNorm): The Folding Operator

Let $x \in \mathbb{R}^D$. For clarity, we define the standard LayerNorm (without learned affine parameters) which projects $x$ via mean $\mu = \frac{1}{D}\mathbf{1}^T x$ and variance $\sigma^2 = \frac{1}{D}\|x - \mu \mathbf{1}\|_2^2$:


$$ y = \frac{x - \mu \mathbf{1}}{\sigma} $$

We take the Jacobian $J_{LN} = \frac{\partial y}{\partial x}$. Using the quotient rule:


$$ \frac{\partial y}{\partial x} = \frac{1}{\sigma} \frac{\partial (x - \mu \mathbf{1})}{\partial x} + (x - \mu \mathbf{1}) \left( \frac{\partial (\sigma^{-1})}{\partial x} \right) $$

1. $\frac{\partial (x - \mu \mathbf{1})}{\partial x} = I - \frac{1}{D}\mathbf{1}\mathbf{1}^T$.
2. $\frac{\partial (\sigma^2)}{\partial x} = \frac{2}{D}(x - \mu \mathbf{1})^T$.
3. $\frac{\partial (\sigma^{-1})}{\partial x} = -\frac{1}{2}(\sigma^2)^{-3/2} \frac{\partial (\sigma^2)}{\partial x} = -\frac{1}{D \sigma^3} (x - \mu \mathbf{1})^T$.

Combining these:


$$ J_{LN} = \frac{1}{\sigma} \left[ \left( I - \frac{1}{D}\mathbf{1}\mathbf{1}^T \right) - \frac{(x - \mu \mathbf{1})(x - \mu \mathbf{1})^T}{D \sigma^2} \right] $$

Let the matrix in the brackets be denoted as $P_{\perp}$. Notice that $(x - \mu \mathbf{1})^T (x - \mu \mathbf{1}) = D \sigma^2$. Therefore, $P_{\perp}$ is symmetric and idempotent ($P_{\perp}^2 = P_{\perp}$). It is an exact orthogonal projection matrix.

The perturbation after normalization is:


$$ \delta y = \frac{1}{\sigma} P_{\perp} \delta x $$

**Analysis of Folding:**
Because $P_{\perp}$ is an orthogonal projection, its spectral norm is exactly $\|P_{\perp}\|_2 = 1$. Therefore:


$$ \|\delta y\|_2 \le \frac{1}{\sigma} \|\delta x\|_2 $$

1. **Orthogonal Annihilation:** $P_{\perp} (x - \mu \mathbf{1}) = 0$. Any component of the perturbation $\delta x$ that points in the radial direction of the centered activation is mathematically annihilated.
2. **Radial Damping:** If the preceding MLP or Attention layers stretch $x$, the absolute variance $\sigma$ increases. This directly divides the perturbation magnitude, applying a strict geometric brake on exponential divergence. This forms the "fold" that maps expanding trajectories back onto a bounded spherical manifold.

---

### 5. The KV Cache: Delay-Coordinate Transmission

To explain the step-function divergence observed in your experiments, we must model the autoregressive loop.

At generation step $t$, the hidden state of the current token $h_t$ is computed by attending to the KV cache of all past tokens $j \le t$:


$$ h_t = \sum_{j=0}^{t} A_{t,j} V_j $$

If we perturb the initial prompt embedding by $\delta x_0$, this perturbation is explicitly stored in the cache as $\delta V_0$. Assuming the generated tokens up to step $t$ have not flipped (i.e., the discrete text is identical), the input at step $t$ is identically unperturbed ($dx_t = 0$), so $A_{t,j}$ and $Q_t$ contain only infinitesimal deviations.

The perturbation at step $t$ is thus governed by:


$$ \delta h_t = \sum_{j=0}^{t} \left[ (\delta A_{t,j}) V_j + A_{t,j} (\delta V_j) \right] $$

**Analysis of the Jump:**
This is a discrete delay-differential equation. Because $\delta V_0 \neq 0$, the sum explicitly injects the historical error into the present state $h_t$ at every generation step. The perturbation $\delta h_t$ grows continuously in the latent space due to the residual spectrum shifting and attention/MLP stretching mechanisms detailed above. However, the argmax selection function filters this latent divergence. The text trajectory only visibly diverges when the latent error $\delta h_t$ projects onto the final unembedding matrix $W_U$ with sufficient magnitude to overcome the logit margin $(z_{top} - z_{runner\_up})$.