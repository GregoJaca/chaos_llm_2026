import sys
sys.path.insert(0, "analysis_update")
from looputil import pairs, find_dirs, model_tags, replay_path, loop_path
import os
print("dirs:", [d.replace(os.getcwd(), "") for d in find_dirs(model_tags("DeepSeek-1.5B"))])
for pl, pr in pairs("DeepSeek-1.5B"):
    print("pair:", pl.replace(os.getcwd(), ""), pr.replace(os.getcwd(), ""))
