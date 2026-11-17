import os, glob, numpy as np, torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer
ROOT = os.path.expanduser('~/LLM/chaos_llm_2026')
snap = glob.glob(os.path.expanduser('~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/*/'))[0]
tok = AutoTokenizer.from_pretrained(snap, local_files_only=True)
text = open(os.path.join(ROOT, 'prompts.txt')).read().split('\t', 1)[1].strip()
ids = tok(text, return_tensors='pt').input_ids.cuda()
out = {}
for dt in [torch.bfloat16, torch.float16]:
    model = AutoModelForCausalLM.from_pretrained(snap, torch_dtype=dt, local_files_only=True,
                                                 attn_implementation='sdpa').cuda().eval()
    model.config.sliding_window = 100; model.config.use_sliding_window = True
    emb = model.get_input_embeddings()(ids)
    def gen(E, n=120):
        am = torch.ones(E.shape[0], E.shape[1], dtype=torch.long, device='cuda')
        with torch.no_grad():
            o = model.generate(inputs_embeds=E, attention_mask=am, max_new_tokens=n,
                               do_sample=False, eos_token_id=None, pad_token_id=0)
        return o.cpu().numpy()
    b1 = gen(emb); b1b = gen(emb); b16 = gen(emb.expand(16, -1, -1))
    mism = [int(np.where(b16[i] != b1[0])[0][0]) if np.any(b16[i] != b1[0]) else -1 for i in range(4)]
    same = [int(np.where(b16[i] != b16[0])[0][0]) if np.any(b16[i] != b16[0]) else -1 for i in range(1, 5)]
    out[str(dt)] = dict(batch1_repeat_identical=bool(np.array_equal(b1[0], b1b[0])),
                        batch16_vs_batch1_first=mism, batch16_rows_vs_row0=same)
    print(dt, out[str(dt)], flush=True)
    del model; torch.cuda.empty_cache()
with open(os.path.join(ROOT, 'analysis_audit', 'ds_batch_audit.json'), 'w') as f:
    json.dump(out, f, indent=2)
