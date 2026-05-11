# Chaos LLM Experiment

This project generates equidistant perturbations in prompt embedding space, runs greedy decoding with optional adaptive early stop, and saves token-id outputs for downstream analysis.

## Setup

Create a virtual environment and install dependencies:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux/HPC:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Edit config.yaml. All parameters are configurable there, including:
- model path
- perturbation magnitudes
- sliding window sizes
- max generation length
- adaptive stop

## Prompt file format

prompts.txt uses one prompt per line. You can optionally name a prompt by prefixing with a name and a tab:

```
my_prompt_name<TAB>This is the prompt text.
```

If no name is provided, a name like prompt_0 is assigned.

## Run

From the repo root:

```
set PYTHONPATH=src
python -m chaos_llm.run_experiment --config config.yaml
```

Linux:

```
export PYTHONPATH=src
python -m chaos_llm.run_experiment --config config.yaml
```

## Outputs

Each run creates a folder named:

```
run_{sliding_window}_{perturbation}_{prompt_name}
```

Inside each run folder:
- config.json: snapshot of the parameters used
- tokens.npz: baseline and perturbed token id sequences

tokens.npz contains:
- baseline_ids: 1D int array (prompt + generated tokens)
- perturbed_ids: 2D int array with padding
- perturbed_lengths: 1D int array of true lengths
- divergence_index: 1D int array, -1 if no divergence before stop
- prompt_len: scalar int

## Notes

- Equidistant perturbations are constructed as a regular simplex in a subspace. This requires num_conditions <= subspace_dim.
- Adaptive stop compares each generated token to the baseline and stops at the first mismatch.
