import numpy as np, sys, os
p = sys.argv[1] if len(sys.argv) > 1 else 'results/big/long4000test/a1.npz'
if not os.path.exists(p):
    print('MISSING', p); sys.exit(1)
d = np.load(p)
print('file', p)
print('T', d['gen_ids'].shape[0], 'radii', d['radii'], 'layers', d['layers'])
a = d['dh_norm'][0, 0, -1, :]
print('dh last layer: median %.4g, max %.4g, nonzero frac %.3f' % (np.median(a), a.max(), (a > 0).mean()))
print('margins: min %.4g, n<0.05 %d' % (d['margins'].min(), (d['margins'] < 0.05).sum()))
print('dm dir0: max|dm| %.4g' % np.abs(d['dm'][0, 0]).max())
