import json

d = json.load(open('shape1.json'))
n = json.load(open('numbers_shape.json'))
for name in ['launch', 'interstellar']:
    print('====', name)
    for lbl, v in d[name]['fig2'].items():
        print('  %-32s IQR/med=%.3f  med=%.4g  slope_range=[%.2f, %.2f]' % (
            lbl, v['iqr_over_med_dist'], v['median_dist'], v['slope_min'], v['slope_max']))
print()
for name in ['interstellar', 'launch', 'cyber']:
    c = n[name]['cross']
    print(name, 'C=0.05:', c['0.05'], ' C=0.1:', c['0.1'], ' rise .1-.9:', round(n[name]['rise_width_dec'], 3))
    print('   ER-legal window:', n[name]['er_legal_window'])
    print('   best 1-dec fit:', n[name]['best_1dec_fit'])
