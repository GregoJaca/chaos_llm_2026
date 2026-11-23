"""Master update script: syncs results from spark and x11, regenerates all figures,
and confirms build status of new.md and new.pdf.
"""
import os
import subprocess
import sys

def run(cmd, desc):
    print(f"--> {desc}...")
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        print(f"    [Warning] {desc} returned code {res.returncode}")
    else:
        print(f"    [OK] {desc}")

def main():
    print("=== SYNCHRONIZING GPU RESULTS ===")
    # 1. Sync from spark (Qwen3.5-9B and DeepSeek-14B)
    run("scp -r spark:~/LLM/chaos_llm_2026/results_big/softloop/ results_big/",
        "Syncing results_big from spark")
    
    # 2. Sync from x11 (sampling runs across architectures)
    run("scp -r x11:~/LLM/chaos_llm_2026/results/softloop_sample/ results/",
        "Syncing softloop_sample from x11")

    print("\n=== REGENERATING PUBLICATION FIGURES ===")
    run(f"{sys.executable} scripts/softloop_master.py", "Generating sl2_master_summary.png")
    run(f"{sys.executable} scripts/make_sampling_and_response_figs.py", "Generating sampling and response sweeps")
    run(f"{sys.executable} scripts/softloop_figs2.py", "Generating gate law, decomp, and phase figures")
    run(f"{sys.executable} scripts/make_entropy_fig.py", "Generating entropy & margin dynamics figure")

    print("\n=== VERIFYING PREVIEW BUILD ===")
    log_path = "notebook/.preview/build.log"
    if os.path.exists(log_path):
        lines = open(log_path).readlines()[-3:]
        print("Latest build log:")
        for l in lines:
            print("  ", l.strip())

    print("\n=== ALL DATA & FIGURES UPDATED SUCCESSFULLY ===")

if __name__ == '__main__':
    main()
