# SumForU

SumForU is an end‑to‑end research project for building persona-aware product summaries. The codebase turns large collections of customer reviews into tailored briefings and suitability ratings that help a specific shopper profile decide whether a product is a good fit. The project leans on the [Tinker](https://tinker-docs.thinkingmachines.ai/) platform for model training and inference, combines supervised fine-tuning with reinforcement learning from AI feedback (RLAIF), and ships reusable tooling for data synthesis, evaluation, and experiment tracking.


## Repository Layout

```
SumForU/
|
|-- dataset/
|   |-- __init__.py                 # Path helpers (RAW/INTERIM/SFT/RL/EVAL)
|   |-- preprocess.py               # Persona extraction from raw review dumps
|   |-- synthesize_data.py          # CLI to build SFT and RL datasets
|   `-- data/
|       |-- raw/                    # Source review corpora and persona seeds
|       `-- processed/
|           |-- sft/                # Synthetic conversations for fine-tuning
|           `-- rl/                 # Persona prompts + rubrics for RLAIF
|
|-- results/
|   |-- v1_test_baseline.json       # Example evaluation dumps
|   |-- v1_test_sft.json            # ...
|   `-- sft_personalized_model/     # Checkpoint metadata + wandb logs
|
|-- scripts/
|   |-- test/
|   |   |-- config.py               # Prompt templates and model registry
|   |   |-- test.py                 # Async inference harness
|   |   `-- utils.py                # Sampler wrapper + prompt builders
|   `-- train/
|       |-- prometheus_types.py     # Prometheus Eval data classes and renderers
|       |-- rl_env.py               # Environment builders for RL preference loops
|       |-- rl.py                   # RLAIF training entry point
|       `-- sft.py                  # Supervised fine-tuning entry point
|
`-- README.md
```


## Quickstart

### 1. Clone & Environment

```bash
git clone git@github.com:Harry20030331/SumForU.git
cd SumForU

# Recommended: create a dedicated virtual environment
conda create -n sumforu python=3.11
conda activate sumforu
```

Install the required Python packages:

```bash
pip install tinker
pip install git+https://github.com/thinking-machines-lab/tinker-cookbook.git
pip install datasets wandb python-dotenv tqdm rich chz numpy torch
```

> **Tinker credentials** – copy `.env.example` to `.env` and add your Tinker API key (and any other secrets your workspace requires). All scripts that talk to Tinker load credentials via `python-dotenv`.

### 2. Prepare Data

Raw review corpora live under `dataset/data/raw/`. If you're starting from raw stringified review dumps, run the persona extraction step first (swap in the paths matching your split):

```bash
python -m dataset.preprocess \
   --input dataset/data/raw/v1_test_stringified.json \
   --output dataset/data/raw/v1_test_preprocessed.json
```

With personas prepared, regenerate the synthetic datasets used for training:

```bash
# Generate SFT conversations from the default preprocessed persona set
python -m dataset.synthesize_data --mode sft \
   --input dataset/data/raw/v1_train_preprocessed.json \
   --sft-output dataset/data/processed/sft/v1_synthesized_output.jsonl

# Generate RL prompts (train/dev) in a separate run
python -m dataset.synthesize_data --mode rl \
   --input dataset/data/raw/v1_train_preprocessed.json \
   --rl-output dataset/data/processed/v1_rl_prompts.jsonl
```

Key artifacts:

- `dataset/data/processed/sft/v1_synthesized_output.jsonl` – conversation-format pairs for SFT.
- `dataset/data/processed/v1_rl_prompts.jsonl` – persona-aligned prompts, rubrics, and optional references for reward modeling / RL.

### 3. Train Models

#### Supervised Fine-tuning (SFT)

`scripts/train/sft.py` wraps the cookbook SFT trainer. Adjust hyperparameters in `build_config()` or override them in a wrapper before launching:

```bash
python -m scripts.train.sft
```

The script reads the SFT JSONL produced in the previous step, logs to wandb, and saves training artifacts under `results/sft_personalized_model/` by default.

#### Reinforcement Learning (RLAIF)

`scripts/train/rl.py` fine-tunes a policy against Prometheus-style pairwise preferences. It defaults to using the same base model for the reward judge; supply `--reward-model-path` if you have a Tinker checkpoint for a reward model.

```bash
python -m scripts.train.rl \
   --train-data-path $(realpath dataset/data/processed/rl/train.jsonl) \
   --test-data-path  $(realpath dataset/data/processed/rl/dev.jsonl) \
   --log-path results/rl_with_rubric_rm \
   --group-size 4 --batch-size 128
```

Arguments map directly to the `main()` signature; run with `--help` to see all available overrides. Training progress, evaluator metrics, and checkpoints are persisted via wandb and the supplied `log_path`.

### 4. Run Evaluations

Use the async tester in `scripts/test/test.py` to compare the baseline model, the SFT checkpoint, or the RL policy on shared prompts.

```bash
python -m scripts.test.test \
   --model_type sft \
   --target_file results/v1_test_sft.json \
   --active_mode json \
   --user_input dataset/data/raw/v1_test_preprocessed.json
```

Tweak `scripts/test/config.py` to point at different checkpoints, prompts, or prompting styles (JSON batch vs. direct single prompt, optional system prompts, sampling hyperparameters).

## Data & Model Pipeline

1. **Persona synthesis** – `dataset/preprocess.py` ingests grouped product reviews and uses a teacher model to write a single-sentence persona focused on purchase behavior.
2. **Conversation generation** – `dataset/synthesize_data.py` rebuilds prompts that pair each persona with review snippets, asks the teacher model for personalized summaries, and emits chat-format records for SFT.
3. **RL prompt curation** – the same script assembles persona prompts, evaluation rubrics, and (optional) human references into JSONL files consumable by the RL environment.
4. **SFT training** – fine-tune a policy on the synthetic conversations with `scripts/train/sft.py` to produce a product-summary specialist.
5. **Rewarded fine-tuning** – run `scripts/train/rl.py` to push the policy further with RLAIF, using the Prometheus-compatible helpers in `scripts/train/rl_env.py` and `scripts/train/prometheus_types.py`.
6. **Evaluation** – inspect responses with the tester, export JSON logs to `results/`, and review wandb dashboards for quantitative trends.

## Experiment Tracking & Outputs

- **W&B** – both SFT and RL scripts log metrics, gradients, and checkpoint metadata to Weights & Biases. Authenticate with `wandb login` before launching training.
- **`results/` directory** – contains example inference dumps (`v1_test_baseline.json`, `v1_test_sft.json`) and the `sft_personalized_model/` wandb run folder for reference.
- **Notebook artifacts** – `tinker_personalization.ipynb` and `tinker_personalization-done.ipynb` capture exploratory analysis, prompt iterations, and rubric experiments.

## Contributing

1. Open an issue describing the feature or fix you plan to make.
2. Create a branch off `main`, implement the change, and add or update documentation as needed.
3. Run the evaluation script or relevant unit tests to validate your work.
4. Submit a pull request with a concise summary and screenshots or logs where helpful.

## Acknowledgements

- The [Tinker Cookbook](https://tinker-docs.thinkingmachines.ai/) for providing the training primitives used throughout the project.
- The Prometheus Eval dataset for rubric-grounded preference modeling inspiration.
- Everyone who contributed anonymized review data powering SumForU’s personas.
