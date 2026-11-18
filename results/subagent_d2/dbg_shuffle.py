import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

X = load_traj('cyber')
scale = median_pairwise_dist(X)
eps = np.geomspace(0.004 * scale, 1.6 * scale, 48)

C1, np1 = c_curve(X, eps, theiler=0)
C2, np2 = c_curve(X.copy(), eps, theiler=0)
Xs = time_shuffle(X)
C3, np3 = c_curve(Xs, eps, theiler=0)
Xs2 = time_shuffle(X)
C4, _ = c_curve(Xs2, eps, theiler=0)

ok = (C1 > 0) & (C3 > 0)
print('npairs', np1, np2, np3)
print('max |logC(X) - logC(copy)|:', np.max(np.abs(np.log(C1[ok]) - np.log(C2[ok]))))
print('max |logC(X) - logC(shuf)| :', np.max(np.abs(np.log(C1[ok]) - np.log(C3[ok]))))
print('max |logC(shuf) - logC(shuf2)|:', np.max(np.abs(np.log(C3[ok]) - np.log(C4[ok]))))
i = np.argmax(np.abs(np.log(C1[ok]) - np.log(C3[ok])))
ii = np.where(ok)[0][i]
print('worst eps idx', ii, 'eps/scale', eps[ii] / scale, 'C1', C1[ii], 'C3', C3[ii])
print('C1[::8]', C1[::8].round(5))
print('C3[::8]', C3[::8].round(5))
