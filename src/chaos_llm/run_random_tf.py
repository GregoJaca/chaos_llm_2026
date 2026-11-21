"""Build a randomly initialized Qwen2 transformer with the exact
DeepSeek-R1-Distill-Qwen-1.5B architecture and a weight-gain knob.

Mean-field theory predicts an order-to-chaos transition as a function of the
scale of the residual-stream updates.  The knob is the init std of the two
matrices that write into the residual stream (attn out_proj, mlp down_proj):
std = 0.02 * gain.  Everything else stays at the Qwen2 default init
(normal(0, 0.02)); norms are 1, biases 0.

The model is saved as a normal HF checkpoint so that run_a1 / run_thresholds /
run_tf_predict work on it unchanged.  The tokenizer files are copied from the
DeepSeek checkpoint (same vocab).

Usage:
  python -m chaos_llm.run_random_tf --gain 2.0 --out ~/LLM/models--randqwen-g2.0
"""
import argparse
import glob
import os
import shutil

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

REF = "~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B"


def build(gain, seed, out):
    torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=151936, hidden_size=1536, intermediate_size=8960,
        num_hidden_layers=28, num_attention_heads=12, num_key_value_heads=2,
        max_position_embeddings=4096, rms_norm_eps=1e-6, tie_word_embeddings=False,
        sliding_window=None, use_sliding_window=False,
    )
    model = Qwen2ForCausalLM(cfg)
    n_upd = n_other = n_emb = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith(".bias"):
                p.zero_()
            elif "norm" in name:
                p.fill_(1.0)
            elif "embed_tokens" in name or "lm_head" in name:
                p.normal_(0.0, 0.02)
                n_emb += p.numel()
            elif name.endswith("out_proj.weight") or name.endswith("down_proj.weight"):
                p.normal_(0.0, 0.02 * gain)
                n_upd += p.numel()
            else:
                p.normal_(0.0, 0.02)
                n_other += p.numel()
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    # tokenizer: copy from the DeepSeek checkpoint (same vocab/arch family).
    # The reference dir may be an HF cache dir with a snapshots/<hash>/ layer.
    import chaos_llm.protocol as protocol
    ref = protocol.resolve_model_path(REF)
    for pat in ["tokenizer*", "vocab.json", "merges.txt", "special_tokens_map*",
                "added_tokens*", "tokenizer_config*"]:
        for f in glob.glob(os.path.join(ref, pat)):
            if os.path.isfile(f):
                shutil.copy(f, os.path.join(out, os.path.basename(f)))
    print("gain=%g seed=%d -> %s" % (gain, seed, out))
    print("params: embed %.0fM update-matrices %.0fM other %.0fM" % (
        n_emb / 1e6, n_upd / 1e6, n_other / 1e6))
    total = (n_emb + n_upd + n_other) / 1e9
    print("total %.2fB params, fp32 on disk %.1f GB" % (total, total * 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gain", type=float, required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    build(args.gain, args.seed, os.path.expanduser(args.out))


if __name__ == "__main__":
    main()
