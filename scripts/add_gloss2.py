p = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\notebook.md"
s = open(p, encoding="utf-8").read()
anchor = "\n---\n\n# Part 1. Definitions"
new = r"""
## 0.5 Geometry terms (used in Part 8 and Part 9)

| term | meaning |
|---|---|
| **SVD (singular value decomposition)** | writing a matrix $M=U\Sigma V^\top$ with $\Sigma$ diagonal and non-negative. The diagonal entries $\sigma_i$ are the singular values: the factors by which $M$ stretches its principal directions. |
| **$\sigma_1$ / $\sigma_{\max}$** | the largest singular value: the maximum stretch over *all* unit vectors. It is attained by one particular direction (the top right-singular vector). It is not the stretch of a generic vector. |
| **rank / effective rank** | the number of non-zero singular values is the rank; the *effective* rank is how many are needed to reach a given fraction (90% here) of the total squared singular values. Used for matrices that are technically full rank but numerically low-rank. |
| **participation ratio (PR)** | $(\sum_i\lambda_i)^2/\sum_i\lambda_i^2$ with $\lambda_i$ the eigenvalues of a covariance. It is an "effective number of dimensions": 1 if all variance is in one direction, $D$ if it is spread evenly over $D$ directions. |
| **stable rank** | $\sigma_1^2/\sum_i\sigma_i^2$, between 1 and the rank; another effective-dimension measure, less sensitive to the tail. |
| **intrinsic dimension (ID)** | the dimension of the manifold the data locally lies on, estimated from nearest-neighbour distances (Levina–Bickel maximum-likelihood estimator, "MLE", with $k$ neighbours). Reported per trajectory because pooling different trajectories inflates/deflates it. |
| **anisotropy** | the mean cosine between two randomly chosen state vectors. 0 = directions are unrelated (isotropic); 1 = all vectors point the same way. |
| **cone** | $\|\mu\|/\overline{\|h\|}$ with $\mu$ the mean state: 0 = the states surround the origin, 1 = they all lie on one ray. |
| **principal angle** | the angle between two subspaces; the cosines of the principal angles are the singular values of the product of their orthonormal bases. Used to ask whether two subspaces (e.g. the top singular directions of consecutive blocks) point the same way. |
| **cosine** between vectors | $\langle a,b\rangle/(\|a\|\|b\|)$: 1 = same direction, 0 = perpendicular. |
| **residual write** | the increment $h_{\ell+1}-h_\ell$ that a block adds to the residual stream. |
| **flux tube** | our name for the picture that emerges: the perturbation stays confined (its norm does not grow) and travels along the trajectory, leaving only at isolated "gates" where the decision margin crosses zero. It is a description, not a derived object. |
| **participation ratio vs ID** | PR/effective rank describe the *linear* spread of the data (covariance); ID describes its *local* manifold dimension. They answer different questions and need not agree. |
"""
assert anchor in s
s = s.replace(anchor, new + anchor, 1)
open(p, "w", encoding="utf-8").write(s)
print("ok")
