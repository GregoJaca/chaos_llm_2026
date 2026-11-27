import numpy as np
rng = np.random.default_rng(0)
D, k = 50, 10
Q1 = np.linalg.qr(rng.normal(size=(D, k)))[0]
Q2 = np.linalg.qr(rng.normal(size=(D, k)))[0]
Q3 = np.linalg.qr(np.column_stack([Q1[:, :5], rng.normal(size=(D, 5))]))[0]


def theta(A, B, kk):
    C = A[:, :kk].T @ B[:, :kk]
    sv = np.linalg.svd(C, compute_uv=False)
    return np.degrees(np.arccos(np.clip(sv.min(), -1, 1)))


print("same subspace  :", [round(theta(Q1, Q1, kk), 3) for kk in (1, 3, 10)])
print("random subspaces:", [round(theta(Q1, Q2, kk), 2) for kk in (1, 3, 10)])
print("half shared    :", [round(theta(Q1, Q3, kk), 2) for kk in (1, 3, 10)])

d = Q1[:, 0] * 3 + Q1[:, 1] * 2
for kk in (1, 3, 10):
    B = Q1[:, :kk]
    p = B @ (B.T @ d)
    print("align k=%2d:" % kk, round(float(np.linalg.norm(p) / np.linalg.norm(d)), 5))
