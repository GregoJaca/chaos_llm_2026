import json
import numpy as np

d = json.load(open('shape2.json'))

def fmt(x, nd=2):
    if x is None:
        return 'None'
    return ('%.' + str(nd) + 'f') % x

for name in ['launch', 'interstellar']:
    for oname in ['pc1', 'norm', 'stepdist']:
        print('====', name, '/', oname, '(tau=1)')
        for m in [2, 3, 4, 5, 6, 8, 10]:
            k = 'm%d_tau1' % m
            v = d[name]['scan'][oname].get(k)
            if v is None:
                continue
            bf = v['best_1dec_fit_r2']
            bfs = 'None' if bf is None else 'd2=%.2f r2=%.3f eps=[%.3g,%.3g]' % tuple(bf)
            print('  m=%2d N=%d slope@C.1=%s slope@C.5=%s widest_flat=%s dec @slope=%s best1dec: %s' % (
                m, v['N'], fmt(v['slope_at_C.1']), fmt(v['slope_at_C.5']),
                fmt(v['widest_flat_dec']), fmt(v['widest_flat_slope']), bfs))
print()
print('best candidate:', d['launch']['best_candidate'])
print('henon:', d['henon'])
print('lorenz:', d['lorenz'])
print('ar1:', d['ar1'])
print('launch_pc1_ref:', d['launch_pc1_ref'])
print('cyber theiler detail:', json.dumps(d['cyber_raw_theiler_detail'], indent=1))
