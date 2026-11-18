import zipfile, pickletools
p = r'C:\Users\hjaca\research\chaos_llm_2026\results\local_data\pecs\cyber\last.pt'
z = zipfile.ZipFile(p)
d = z.read('hidden_states_last/data.pkl')
pickletools.dis(d[:1500])
