import json
c = json.load(open("/home/grego/LLM/models/phi-4/config.json"))
print("phi4:", {k: c.get(k) for k in ["architectures", "hidden_size", "num_hidden_layers",
                                      "intermediate_size", "vocab_size", "sliding_window",
                                      "max_position_embeddings"]})
c = json.load(open("/home/grego/LLM/models/Qwen3-14B/config.json"))
print("q3_14b:", {k: c.get(k) for k in ["architectures", "hidden_size", "num_hidden_layers",
                                        "intermediate_size", "vocab_size", "sliding_window"]})
