import numpy as np, os
BASE = r'C:\Users\hjaca\research\chaos_llm_2026\results\local_data\tf_pred'
for name in ['mid', 'large']:
    d = np.load(os.path.join(BASE, name + '.npz'), allow_pickle=True)
    pred, ar, ref, radii = d['pred'], d['ar'], d['ref'], d['radii']
    print('==', name, radii)
    for ri, r in enumerate(radii):
        ok = []
        Tp_l, Ta_l = [], []
        fp = 0
        for di in range(pred.shape[1]):
            a, p, f = ar[ri, di], pred[ri, di], ref
            v = a >= 0
            df = np.where((a != f) & v)[0]
            dp = np.where((p != f) & v)[0]
            ta = int(df[0]) if len(df) else -1
            tp = int(dp[0]) if len(dp) else -1
            Ta_l.append(ta); Tp_l.append(tp)
            if ta >= 0 and tp >= 0 and tp < len(a):
                ok.append(int(p[tp] == a[ta]))
            if ta < 0 and tp >= 0:
                fp += 1
        Ta = np.array(Ta_l); Tp = np.array(Tp_l)
        both = (Ta >= 0) & (Tp >= 0)
        exact = np.mean((Ta == Tp) | ((Ta < 0) & (Tp < 0)))
        mdt = np.median(np.abs(Ta[both] - Tp[both])) if both.any() else -1
        acc = np.mean(ok) if ok else float('nan')
        print(' r=%-5g flips=%2d predflips=%2d falsepos=%2d step_exact=%.2f token_at_flip=%.2f median|dT|=%.0f'
              % (r, int((Ta >= 0).sum()), int((Tp >= 0).sum()), fp, exact, acc, mdt))
