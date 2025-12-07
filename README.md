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
   --input dataset/data/raw/v1_test_preprocessed.json \
   --sft-output dataset/data/processed/sft/v1_synthesized_test_output.jsonl

# Generate RL prompts (train split)
python -m dataset.synthesize_data --mode rl \
   --input dataset/data/raw/v1_train_preprocessed.json \
   --rl-output dataset/data/processed/rl/v1_rl_train.jsonl

# Optionally, generate RL prompts for a test/dev split
python -m dataset.synthesize_data --mode rl \
   --input dataset/data/raw/v1_test_preprocessed.json \
   --rl-output dataset/data/processed/rl/v1_rl_test.jsonl
```

Key artifacts:

- `dataset/data/processed/sft/v1_synthesized_output.jsonl` – conversation-format pairs for SFT.
- `dataset/data/processed/rl/v1_rl_train.jsonl` – persona-aligned prompts, rubrics, and optional references for reward modeling / RL.
- `dataset/data/processed/rl/v1_rl_test.jsonl` (optional) – run the same RL command against a held-out persona file to produce evaluation prompts.

### 3. Train Models

#### Supervised Fine-tuning (SFT)

`scripts/train/sft.py` wraps the cookbook SFT trainer. Adjust hyperparameters in `build_config()` or override them in a wrapper before launching:

```bash
python -m scripts.train.sft
```

Pass overrides (for example, to tweak the shorter five-sentence prompts we train on) via CLI flags:

```bash
python -m scripts.train.sft \
   --log-path results/logs/sft_4b_v2 \
   --wandb-name sft_4b_v2
```

The default `max_length` is 65536 tokens to match the compact prompt/response pairs; raise it if you extend conversations. The script reads the SFT JSONL produced in the previous step, logs to wandb, and saves training artifacts under `results/sft_personalized_model/` by default.

#### Reinforcement Learning (RLAIF)

`scripts/train/rl.py` fine-tunes a policy against Prometheus-style pairwise preferences. It defaults to using the same base model for the reward judge; supply `--reward-model-path` if you have a Tinker checkpoint for a reward model.

```bash
python -m scripts.train.rl \
   log_path=results/logs/rl_personalized_model \
   wandb_name=rl_personalized_model
```

```bash
python -m scripts.train.rl \
   log_path=results/logs/rl_personalized_model_sftinit_v2 \
   wandb_name=rl_personalized_model_sftinit_v2 \
   learning_rate=5e-6 \
   train_repeat=15 \
   eval_every=10 \
   model_path=tinker://1adb47b4-105b-5a29-96fc-d04511e11a1c:train:0/weights/final
```

Append `test_data_path=$(realpath dataset/data/processed/rl/v1_rl_test.jsonl)` if you generate a held-out split.

Arguments map directly to the `main()` signature; run with `--help` to see all available overrides. Training progress, evaluator metrics, and checkpoints are persisted via wandb and the supplied `log_path`.

### 4. Run Evaluations

Use the async tester in `scripts/test/test.py` to compare the baseline model, the SFT checkpoint, or the RL policy on shared prompts.

```bash
python -m scripts.test.test --model_type rl \
   --target_file results/v1_test_rl.json
```

Tweak `scripts/test/config.py` to point at different checkpoints, prompts, or prompting styles (JSON batch vs. direct single prompt, optional system prompts, sampling hyperparameters).

### 5. Calculate quantitative metric for evaluation
After obtaining model outputs with scripts/test/test.py, use the tools in scripts/eval to compute all quantitative metrics on the test set.

Run the main metric script to get text, semantic, coverage, and rating metrics for all four systems (baseline / PE / SFT / RL):

```bash
python scripts/eval/eval_summaries_multi.py \
  --gt-path dataset/data/raw/v1_test_preprocessed.json \
  --baseline-path results/v1_test_baseline.json \
  --pe-path results/v1_test_pe.json \
  --sft-path results/v1_test_235B.json \
  --rl-path results/v1_test_rl.json
```

This prints:

Text quality & diversity: ROUGE-1/2/L/Lsum, BLEU-4, Distinct-2/3, USR, ENTR

Semantic similarity: BERTScore (P/R/F1)

Persona/content coverage: RevCov (review coverage), PersCov (persona coverage)

Rating alignment: MAE, MSE, Pearson/Spearman, ExactAcc, Within1Acc, MacroF1, BalancedAcc

Optionally, visualize how Suitability behaves as a rating predictor with confusion matrices and per-bin bias:

```bash
python plot_score_analysis.py \
  --gt-path ../../dataset/data/raw/v1_test_preprocessed.json \
  --baseline-path ../../results/v1_test_baseline.json \
  --pe-path ../../results/v1_test_pe.json \
  --sft-path ../../results/v1_test_sft.json \
  --rl-path ../../results/v1_test_rl.json \
  --output-path confusion_and_bias_all.png
```
This generates a single figure stacking all methods, used in the analysis section to inspect over/under‑rating patterns across score bins.


## Experiment Tracking & Outputs

- The [Tinker Cookbook](https://tinker-docs.thinkingmachines.ai/) for providing the training primitives used throughout the project.
- The [SALT-NLP/cs329x_hw2](https://github.com/SALT-NLP/cs329x_hw2) team for open-sourcing prompts and evaluation patterns that informed our baseline experiments.
- Everyone who contributed anonymized review data powering SumForU’s personas.
