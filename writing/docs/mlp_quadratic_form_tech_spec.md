### SYSTEM INSTRUCTION: LLM EXPERIMENTAL SPECIFICATION

**Objective:** Compute the static structural stretching capacity of the SwiGLU MLP blocks across all 28 layers of `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`.
**Environment:** 16GB Tesla T4 GPU, PyTorch.
**Mathematical Target:** Evaluate the bounds and expected values of the quadratic form governing the Frobenius norm of the MLP Jacobian, isolating the input-independent weight matrices ($M$) from the input-dependent activation vectors ($v$).

---

### 1. Mathematical Formulation

For the SwiGLU architecture, the exact MLP Jacobian is:


$$ J_{MLP} = W_{down} \left[ \text{diag}(s) W_{up} + \text{diag}(h_{up} \odot s') W_{gate} \right] $$


where $s = \text{SiLU}(W_{gate} x)$, $s' = \text{SiLU}'(W_{gate} x)$, and $h_{up} = W_{up} x$.

The total phase-space expansion is measured by the Frobenius norm $\|J_{MLP}\|_F^2 = \text{Tr}(J_{MLP} J_{MLP}^T)$. Expanding this trace for SwiGLU yields a composite quadratic form separated into three distinct structural matrices ($M$) and two activation vectors:


$$ \|J_{MLP}\|_F^2 = v_1^T M_{up} v_1 + v_2^T M_{gate} v_2 + 2 v_1^T M_{cross} v_2 $$

**The Activation Vectors (Dynamic, Input-Dependent):**

* $v_1 = s \in \mathbb{R}^{d_{ff}}$
* $v_2 = h_{up} \odot s' \in \mathbb{R}^{d_{ff}}$

**The $M$ Matrices (Static, Input-Independent):**
Let $H_{down} = W_{down}^T W_{down} \in \mathbb{R}^{d_{ff} \times d_{ff}}$. The structural capacity matrices are defined via Hadamard (element-wise) products:

* $M_{up} = H_{down} \odot (W_{up} W_{up}^T)$
* $M_{gate} = H_{down} \odot (W_{gate} W_{gate}^T)$
* $M_{cross} = H_{down} \odot (W_{up} W_{gate}^T)$

### 2. Evaluation Metrics

To evaluate the bounds of this quadratic form without running a forward pass, we extract two scalar metrics from each $M$ matrix:

1. **The Trace ($\text{Tr}(M)$):**
* **Definition:** The sum of the diagonal elements.
* **Meaning:** Represents the *expected* structural stretching capacity. If the activation vectors $v$ are modeled as isotropic random variables with variance $\sigma^2$, the expected Jacobian norm scales proportionally to the trace.


2. **The Maximum Eigenvalue ($\lambda_{max}(M)$):**
* **Definition:** The spectral radius of the symmetric matrix.
* **Meaning:** By the Rayleigh-Ritz theorem, this represents the *absolute upper bound* of the stretching capacity. The quadratic form $v^T M v$ can never exceed $\lambda_{max}(M) \|v\|_2^2$.



---

### 3. Execution Algorithm (Memory-Safe for 16GB VRAM)

The intermediate dimension is $d_{ff} = 8960$. A single dense float32 matrix of size $8960 \times 8960$ consumes approximately 321 MB of VRAM. Attempting to compute or store all layers simultaneously will result in an Out-Of-Memory (OOM) error.

**Loop Execution per Layer (1 to 28):**

1. **Load Weights to GPU:**
* Extract $W_{up}$, $W_{gate}$, and $W_{down}$ for the current layer $\ell$.
* Shape of $W_{up}, W_{gate}$: `[8960, 1536]`.
* Shape of $W_{down}$: `[1536, 8960]`.


2. **Compute Gramians:**
* $H_{down} = W_{down}^T \times W_{down}$
* $H_{up} = W_{up} \times W_{up}^T$
* $H_{gate} = W_{gate} \times W_{gate}^T$
* $H_{cross} = W_{up} \times W_{gate}^T$


3. **Compute Hadamard Products:**
* $M_{up} = H_{down} \odot H_{up}$
* $M_{gate} = H_{down} \odot H_{gate}$
* *Note: $M_{cross}$ is asymmetric, so computing its exact eigenvalue requires care. Focus eigenvalues strictly on the symmetric $M_{up}$ and $M_{gate}$.*


4. **Extract Trace:**
* Compute and store $\text{Tr}(M_{up})$ and $\text{Tr}(M_{gate})$.
* *Optimization:* The trace of a Hadamard product $A \odot B$ is simply the dot product of their diagonals. You do not need to compute the full dense $M$ matrices just for the trace: `trace = (diag(H_down) * diag(H_up)).sum()`.


5. **Extract Maximum Eigenvalues:**
* Do **NOT** use `torch.linalg.eigh` on the full $8960 \times 8960$ matrix, as the full eigendecomposition is $O(N^3)$ and computationally prohibitive.
* Use **Lanczos Iteration / Power Iteration** to find just the largest eigenvalue. `scipy.sparse.linalg.eigsh(M, k=1, which='LA')` or a custom PyTorch power iteration loop for 100 steps.


6. **Free Memory:**
* Explicitly `del` all $H$ and $M$ tensors.
* Call `torch.cuda.empty_cache()` before moving to layer $\ell+1$.



---

### 4. Interpretation Guide

Once the script outputs the Trace and $\lambda_{max}$ for all 28 layers, evaluate the bounds against the ambient dimension $d = 1536$:

* **The Baseline Threshold:** For the MLP to be a net stretcher of the phase space, the average singular value of the Jacobian must exceed 1. This requires $\|J_{MLP}\|_F^2 > 1536$.
* **Evaluating the Upper Bound:** If $\lambda_{max}(M_{up})$ and $\lambda_{max}(M_{gate})$ are significantly greater than 1536 (e.g., $10^4$ or $10^5$), you have mathematically proven that the structural weights possess the *capacity* to induce chaotic stretching, provided the activation vectors $v$ align with the principal eigenvectors.
* **Evaluating the Expected Stretch:** If $\text{Tr}(M) \gg 1536$, it proves that the matrices are tuned such that even a completely random, isotropic activation pattern will result in net phase-space expansion.