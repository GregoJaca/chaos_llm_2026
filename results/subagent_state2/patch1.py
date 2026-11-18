p = r'C:\Users\hjaca\research\chaos_llm_2026\results\subagent_state2\s1_decay.py'
lines = open(p, encoding='utf-8').read().splitlines()
assert lines[36].strip().startswith('p0, lo, hi') and lines[37].strip().startswith('[1e-12'), lines[36:38]
lines[36] = ("            p0, lo, hi = [y[0] * 0.6, min(T0, 25), y[0] * 0.5, max(T0 * 8, 40)], "
             "[1e-12, 0.3, 1e-12, 3.0], [np.inf, 30, np.inf, 3000]")
lines[37] = ""
open(p, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print('done')
