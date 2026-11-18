import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from pt_load import load_pt

base = r'C:\Users\hjaca\research\chaos_llm_2026\results\local_data\pecs'
paths = {
    'cyber': base + r'\cyber\last.pt',
    'future': base + r'\future\last.pt',
    'great': base + r'\great\last.pt',
    'interstellar': base + r'\interstellar\last.pt',
    'first_layer': base + r'\hidden_states_first.pt',
    'cond0': r'C:\Users\hjaca\research\chaos_llm_2026\results\local_data\pecs\..\launch\hidden_states_cond_0_layer_-1.pt',
}
for k, p in paths.items():
    try:
        a = load_pt(p)
        print(k, a.shape, a.dtype, 'nan:', np.isnan(a).sum(), 'mean|.|:', np.abs(a).mean())
    except Exception as e:
        print(k, 'ERR', repr(e)[:300])
