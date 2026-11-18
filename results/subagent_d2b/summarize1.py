import json

d = json.load(open('shape1.json'))
for name in ['interstellar', 'launch']:
    v = d[name]
    print('====', name, 'scale=%.4g' % v['scale'])
    print('  fig1 identity: center-vs-raw f64 dlogC=%.2e ; fp32-vs-f64 dlogC=%.2e' % (
        v['fig1']['max_dlogC_center_vs_raw_f64'], v['fig1']['max_dlogC_f32raw_vs_f64center']))
    for lbl, vv in v['fig2'].items():
        wf = vv.get('widest_flat')
        s1 = vv.get('slope_at_C0.1')
        s5 = vv.get('slope_at_C0.5')
        s1s = 'nan' if s1 is None else '%.2f' % s1
        s5s = 'nan' if s5 is None else '%.2f' % s5
        if wf:
            wfs = 'slope=%.2f width=%.2fdec eps=[%.3g,%.3g]xmed C=[%.2g,%.2g]' % (
                wf['slope'], wf['width_dec'], wf['eps0'] / vv['median_dist'],
                wf['eps1'] / vv['median_dist'], wf['C0'], wf['C1'])
        else:
            wfs = 'NONE'
        print('   fig2 %-32s med=%.4g IQR/med=%.3f slope@C.1=%s slope@C.5=%s slope_range=[%.2f,%.2f] widest_flat: %s' % (
            lbl, vv['median_dist'], vv['iqr_over_med_dist'], s1s, s5s,
            vv['slope_min'], vv['slope_max'], wfs))
    print('  fig4 theiler:', json.dumps(v['fig4_theiler']))
print('==== cyber scale=%.4g' % d['cyber']['scale'], 'tube d2=%.3f r2=%.3f' % tuple(d['cyber']['tube_fit']),
      'ring d2=%.3f r2=%.4f' % tuple(d['cyber']['ring_fit']))
print('==== cyber fig4:', json.dumps(d['cyber']['fig4_theiler']))
print('==== first scale=%.4g' % d['first']['scale'], 'dup_floor_C=%.4f' % d['first']['dup_floor_C'])
print('==== launch selfpair floor=%.3g ; lin-grid C@0.03med=%.3g' % (
    d['launch']['selfpair_floor'], d['launch']['lin_grid_C_at_0.03med']))
