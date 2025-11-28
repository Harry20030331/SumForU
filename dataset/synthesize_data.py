"""Utility script to synthesize datasets for the persona summary task.

This module centralizes the data generation pipeline so we can create either:
- Supervised fine-tuning (SFT) conversation data with summaries written
    by a stronger teacher model (defaults to Qwen/Qwen3-235B-A22B-Instruct-2507), or
- Reinforcement Learning from AI Feedback (RLAIF) preference prompts that are
    compatible with ``scripts.train.rl``.

Usage (examples):

        # Generate SFT data using default paths
        python -m dataset.synthesize_data

        # Generate only RL prompts (train/dev), keeping existing SFT JSONL
        python -m dataset.synthesize_data --mode rl --rl-output dataset/data/processed/rl

        # Limit to first 200 persona entries and use a smaller validation split
        python -m dataset.synthesize_data --mode rl --max-prompts 200 --dev-count 100

The script assumes the preprocessed persona dataset lives at
``dataset/data/interim/v1_preprocessed.json`` and will write outputs to
``dataset/data/processed`` by default. Use CLI flags if you need different
paths.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from dataset import INTERIM_DIR, RL_DIR, SFT_DIR
from scripts.test import config
from scripts.test.utils import TinkerSampler, build_user_prompt, clear_content
from tinker_cookbook import renderers

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_INPUT_PATH = INTERIM_DIR / "v1_preprocessed.json"
DEFAULT_SFT_OUTPUT = SFT_DIR / "v1_synthesized_output.jsonl"
DEFAULT_RL_OUTPUT = RL_DIR
DEFAULT_RUBRIC = (
    "Does the assistant deliver a concise 2-3 sentence summary that focuses on "
    "evidence from the reviews most relevant to the persona, and provide a "
    "1-10 suitability rating with a short justification grounded in repeated "
    "customer feedback?"
)
DEFAULT_MODEL_NAME = "Qwen/Qwen3-235B-A22B-Instruct-2507"
REFERENCE_PROMPT_SUFFIX = (
    "\n\nI also provide the real review written by the customer who shares this "
    "persona and purchased the product. Use it strictly as additional context "
    "to understand the persona's priorities; do not copy or mirror the review.\n"
)


@dataclass
class PersonaRecord:
    persona: str
    reviews: str
    reference_output: Sequence[str]

    @classmethod
    def from_dict(cls, payload: dict) -> "PersonaRecord":
        return cls(
            persona=payload.get("persona", ""),
            reviews=payload.get("reviews", ""),
            reference_output=payload.get("reference_output", []) or [],
        )


# ---------------------------------------------------------------------------
# SFT data generation
# ---------------------------------------------------------------------------
async def _synthesize_single_summary(
    sampler: TinkerSampler,
    record: PersonaRecord,
    include_reference: bool = True,
) -> dict:
    """Generate one conversation datum for SFT training."""
    system_prompt = config.SYSTEM_PROMPT.strip()
    user_template = config.USER_PROMPT.strip()
    user_prompt = build_user_prompt(record.persona, record.reviews).strip()

    if include_reference and record.reference_output:
        reference_block = "\n".join(reference.strip() for reference in record.reference_output if reference)
        user_prompt += f"{REFERENCE_PROMPT_SUFFIX}{reference_block}"

    messages = [
        renderers.Message(role="system", content=system_prompt),
        renderers.Message(role="user", content=user_prompt),
    ]

    response = await sampler.generate(messages)
    summary = clear_content(response["content"])

    return {
        "messages": [
            {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"},
            {"role": "assistant", "content": summary},
        ]
    }


async def generate_sft_dataset(
    data: Sequence[PersonaRecord],
    output_path: Path,
    model_name: str = DEFAULT_MODEL_NAME,
    concurrency: int = 8,
) -> None:
    """Generate SFT JSONL by prompting a stronger teacher model."""
    if not data:
        raise ValueError("No persona records provided for SFT synthesis.")

    sampler = TinkerSampler(model_name=model_name, temperature=0.7, max_tokens=512, top_p=0.9)
    semaphore = asyncio.Semaphore(max(concurrency, 1))

    async def bounded(idx: int, record: PersonaRecord) -> tuple[int, dict]:
        async with semaphore:
            synthesized = await _synthesize_single_summary(sampler, record)
            return idx, synthesized

    tasks = [bounded(idx, record) for idx, record in enumerate(data)]
    collected: list[tuple[int, dict]] = []

    for idx, task in enumerate(asyncio.as_completed(tasks), start=1):
        result = await task
        collected.append(result)
        print(f"[SFT] completed {idx}/{len(tasks)} prompts", flush=True)

    collected.sort(key=lambda pair: pair[0])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for _, sample in collected:
            fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"[SFT] Wrote {len(collected)} examples to {output_path}")


# ---------------------------------------------------------------------------
# RL prompt generation
# ---------------------------------------------------------------------------
def build_prompt_conversation(record: PersonaRecord, *, include_system: bool) -> list[dict]:
    messages: list[dict] = []
    if include_system:
        messages.append({"role": "system", "content": config.SYSTEM_PROMPT.strip()})
    user_prompt = build_user_prompt(record.persona, record.reviews)
    messages.append({"role": "user", "content": user_prompt})
    return messages


def generate_rl_dataset(
    data: Sequence[PersonaRecord],
    train_output: Path,
    dev_output: Path,
    rubric: str = DEFAULT_RUBRIC,
    dev_count: int = 500,
    seed: int | None = 42,
) -> None:
    """Write RL prompts for train/dev splits."""
    if not data:
        raise ValueError("No persona records provided for RL synthesis.")

    if seed is not None:
        random.seed(seed)
    shuffled = list(data)
    random.shuffle(shuffled)

    dev_count = max(0, min(dev_count, len(shuffled)))
    dev_records = shuffled[:dev_count]
    train_records = shuffled[dev_count:]

    train_output.parent.mkdir(parents=True, exist_ok=True)
    dev_output.parent.mkdir(parents=True, exist_ok=True)

    with train_output.open("w", encoding="utf-8") as train_f:
        for record in train_records:
            payload = {
                "prompt_conversation": build_prompt_conversation(record, include_system=True),
                "reference": "\n".join(record.reference_output) or None,
                "rubric": rubric,
            }
            train_f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    with dev_output.open("w", encoding="utf-8") as dev_f:
        for record in dev_records:
            payload = {
                "prompt_conversation": build_prompt_conversation(record, include_system=False),
                "reference": "\n".join(record.reference_output) or None,
                "rubric": rubric,
            }
            dev_f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    print(
        f"[RL] Wrote {len(train_records)} train prompts to {train_output} and "
        f"{len(dev_records)} dev prompts to {dev_output}"
    )


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------
def load_persona_records(path: Path) -> list[PersonaRecord]:
    if not path.exists():
        raise FileNotFoundError(f"Persona dataset not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return [PersonaRecord.from_dict(entry) for entry in raw]


def apply_slice(data: Sequence[PersonaRecord], limit: int | None) -> Sequence[PersonaRecord]:
    if limit is None or limit <= 0:
        return data
    return data[:limit]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize datasets for persona personalization")
    parser.add_argument("--mode", choices=("sft", "rl"), default="sft",
                        help="Select which dataset to generate in this run.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--sft-output", type=Path, default=DEFAULT_SFT_OUTPUT)
    parser.add_argument("--rl-output", type=Path, default=DEFAULT_RL_OUTPUT,
                        help="Directory or filename where RL JSONL files will be written.")
    parser.add_argument("--rubric", type=str, default=DEFAULT_RUBRIC)
    parser.add_argument("--dev-count", type=int, default=500)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    records = apply_slice(load_persona_records(args.input), args.max_prompts)
    print(f"Loaded {len(records)} persona records from {args.input}")

    if args.mode == "sft":
        asyncio.run(
            generate_sft_dataset(
                records,
                output_path=args.sft_output,
                model_name=args.model_name,
                concurrency=args.concurrency,
            )
        )

    elif args.mode == "rl":
        rl_base = args.rl_output
        if rl_base.suffix == ".jsonl":
            train_output = rl_base
            dev_output = rl_base.with_name(f"{rl_base.stem}_dev.jsonl")
        else:
            train_output = rl_base / "train.jsonl"
            dev_output = rl_base / "dev.jsonl"

        generate_rl_dataset(
            records,
            train_output=train_output,
            dev_output=dev_output,
            rubric=args.rubric,
            dev_count=args.dev_count,
            seed=args.seed,
        )
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
