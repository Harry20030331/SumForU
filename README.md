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
   --log-path results/logs/sft_personalized_model_v3 \
   --wandb-name sft_personalized_model_v3   \
   --num-epochs 50
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
   log_path=results/logs/rl_personalized_model_sftinit_v4 \
   wandb_name=rl_personalized_model_sftinit_v4 \
   learning_rate=1e-5 \
   train_repeat=15 \
   eval_every=10 \
   model_path=tinker://1d8eac56-400a-5deb-9b66-13796f2089df:train:0/weights/final
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
python -m scripts.eval.eval_summaries_multi \
  --gt-path dataset/data/raw/v1_test_preprocessed.json \
  --b235b-path results/v1_test_235B.json \
  --baseline-path results/v1_test_baseline.json \
  --pe-path results/v1_test_pe.json \
  --sft-path results/v1_test_sft.json \
  --rl-path results/v1_test_rl.json
```

This prints:

Text quality & diversity: ROUGE-1/2/L/Lsum, BLEU-4, Distinct-2/3, USR, ENTR

Semantic similarity: BERTScore (Rev-P/Ref-R/Per-R/Ref-F1)

Persona/content coverage: RevCov (review coverage), PersCov (persona coverage)

Rating alignment: MAE, MSE, Pearson/Spearman, ExactAcc, Within1Acc, MacroF1, BalancedAcc

Optionally, visualize how Suitability behaves as a rating predictor with confusion matrices and per-bin bias:

```bash
python -m scripts.eval.plot_score_analysis \
  --gt-path dataset/data/raw/v1_test_preprocessed.json \
  --baseline-path results/v1_test_baseline.json \
  --pe-path results/v1_test_pe.json \
  --sft-path results/v1_test_sft.json \
  --rl-path results/v1_test_rl.json \
  --output-path confusion_and_bias_all.png
```
This generates a single figure stacking all methods, used in the analysis section to inspect over/under‑rating patterns across score bins.


## Experiment Tracking & Outputs

- The [Tinker Cookbook](https://tinker-docs.thinkingmachines.ai/) for providing the training primitives used throughout the project.
- The [SALT-NLP/cs329x_hw2](https://github.com/SALT-NLP/cs329x_hw2) team for open-sourcing prompts and evaluation patterns that informed our baseline experiments.
- Everyone who contributed anonymized review data powering SumForU’s personas.



=== Text quality, diversity, semantic similarity, and coverage ===

method          r1      r2      rL   rLsum   bleu4     D-2     D-3     USR    ENTR  RevCov PersCov   RefBS-R   RevBS-P  PersBS-R
gt          1.0000  1.0000  1.0000  1.0000  1.0000  0.8271  0.9728  1.0000  6.3820  0.6996  0.0625    1.0000    0.8428    0.8321
235b_ref    0.1610  0.0106  0.0979  0.0977  0.0000  0.7228  0.9053  1.0000  6.2516  0.3947  0.1839    0.8406    0.8416    0.8760
baseline    0.1509  0.0091  0.0977  0.0975  0.0000  0.7725  0.9259  1.0000  6.1601  0.4618  0.2380    0.8367    0.8509    0.8746
pe          0.1528  0.0094  0.0957  0.0956  0.0000  0.7430  0.9140  1.0000  6.2274  0.4759  0.1883    0.8389    0.8495    0.8733
sft         0.1486  0.0080  0.0919  0.0919  0.0000  0.7125  0.8941  1.0000  6.2076  0.4136  0.1768    0.8377    0.8424    0.8738
rl          0.1494  0.0082  0.0867  0.0864  0.0000  0.7089  0.8986  1.0000  6.5932  0.4145  0.1485    0.8383    0.8322    0.8765

BERTScore F1 per method (summary vs reference):
  gt         BERTScore-F1 = 1.0000
  235b_ref   BERTScore-F1 = 0.8381
  baseline   BERTScore-F1 = 0.8390
  pe         BERTScore-F1 = 0.8390
  sft        BERTScore-F1 = 0.8359
  rl         BERTScore-F1 = 0.8271

=== Suitability vs reference rating (0-5 scale) ===

method         MAE     MSE   Pearson  Spearman   ExactAcc   Within1Acc    MacroF1    BalancedAcc
gt          0.0000  0.0000    1.0000    1.0000     1.0000       1.0000     1.0000         1.0000
235b_ref    1.1150  1.8875    0.6431    0.6571     0.1700       0.7700     0.1652         0.3003
baseline    1.6050  3.8225    0.2788    0.2294     0.0700       0.5900     0.0788         0.1737
pe          1.4700  3.0850    0.3631    0.3764     0.1000       0.5900     0.1027         0.2097
sft         1.3700  2.7750    0.4222    0.4230     0.1400       0.6200     0.1266         0.2394
rl          1.3650  3.4075    0.4193    0.4489     0.1500       0.6900     0.1476         0.2404