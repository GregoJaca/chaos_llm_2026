import json

d = json.load(open('shape3.json'))
for key in ['white', 'ar1_tau1', 'lorenz_msweep']:
    print('====', key)
    for m, v in d[key].items():
        print('  m=%2d slope@C.1=%s flat=%s dec @slope=%s' % (
            int(m), ('%.2f' % v['slope_at_C.1']) if v['slope_at_C.1'] else 'None',
            ('%.2f' % v['flat_dec']) if v['flat_dec'] is not None else 'None',
            ('%.2f' % v['flat_slope']) if v['flat_slope'] is not None else 'None'))
print()
print('lorenz plateau:', d['lorenz']['plateau'])
print('henon plateau:', d['henon']['plateau'])
print('case_a:', json.dumps(d['case_a'], indent=1)[:800])
print('case_b slope data:', d['case_b']['slope_at_C.1_data'], 'flat:', d['case_b']['flat_dec'])
for k, v in d['case_b']['surrogates'].items():
    print('  case_b', k, [('%s/%s/%.2f' % (('%.2f' % r['slope_at_C.1']) if r['slope_at_C.1'] else 'None',
                                            ('%.2f' % r['flat_dec']) if r['flat_dec'] is not None else 'None',
                                            r['max_dlogC_vs_data'])) for r in v])
