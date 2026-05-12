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
python run_experiment.py --config config.yaml
```

Linux:
python run_experiment.py --config config.yaml
```

If output.save_text is true, each run also writes a JSON file (see output.text_filename)
with baseline and perturbed decoded text.
```
run_{sliding_window}_{perturbation}_{prompt_name}
```

Inside each run folder:
- config.json: snapshot of the parameters used
- tokens.npz: baseline and perturbed token id sequences

tokens.npz contains:
- baseline_ids: 1D int array (prompt + generated tokens if enabled)
- perturbed_ids: 2D int array with padding
- perturbed_lengths: 1D int array of true lengths
- divergence_index: 1D int array, -1 if no divergence before stop
- prompt_len: scalar int

## Notes

- Equidistant perturbations are constructed as a regular simplex in a subspace. This requires num_conditions <= subspace_dim.
- Adaptive stop compares each generated token to the baseline and stops at the first mismatch.

## Analysis

Edit analysis.yaml, then run:

```
python run_analysis.py --config analysis.yaml

nohup python -m chaos_llm.run_experiment --config config.yaml > run_experiment.out 2>&1 &
```

## Inspect tokens

Print token ids (and optionally token strings) from a run folder:

```
python print_tokens.py --run-dir run_4096_0.0004_interstellar_travel --max-tokens 80 --config config.yaml
```

## Export text without re-running

Decode an existing tokens.npz into a text JSON file:

```
python export_text.py --run-dir run_4096_0.0004_interstellar_travel --config config.yaml
```

With token string decoding:

```
python print_tokens.py --run-dir run_4096_0.0004_interstellar_travel --decode --config config.yaml
```

## Tests

Run the unit tests:

```
python -m unittest tests.test_divergence
python -m unittest tests.test_analysis_e2e
```

## Environment

If you use VS Code, the repo includes a .env file that sets PYTHONPATH=src.
The wrapper scripts above also handle PYTHONPATH automatically.
