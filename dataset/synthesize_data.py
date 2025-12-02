"""Utility script to synthesize datasets for the persona summary task.

This module centralizes the data generation pipeline so we can create either:
- Supervised fine-tuning (SFT) conversation data with summaries written
    by a stronger teacher model (defaults to Qwen/Qwen3-235B-A22B-Instruct-2507), or
- Reinforcement Learning from AI Feedback (RLAIF) preference prompts that are
    compatible with ``scripts.train.rl``.

Usage (examples):

        # Generate SFT data using default paths
        python -m dataset.synthesize_data

        # Generate only RL prompts (train split), keeping existing SFT JSONL
        python -m dataset.synthesize_data --mode rl --rl-output dataset/data/processed/rl/v1_rl_train.jsonl

        # Limit to first 200 persona entries for a quicker RL run
        python -m dataset.synthesize_data --mode rl --max-prompts 200

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
from typing import Sequence

from dataset import INTERIM_DIR, RL_DIR, SFT_DIR
from scripts.test import config
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_INPUT_PATH = INTERIM_DIR / "v1_preprocessed.json"
DEFAULT_SFT_OUTPUT = SFT_DIR / "v1_synthesized_output.jsonl"
DEFAULT_RL_OUTPUT = RL_DIR / "v1_rl_train.jsonl"
DEFAULT_RUBRIC = (
    "Does the assistant deliver a concise 2-3 sentence summary that focuses on "
    "evidence from the reviews most relevant to the persona, and provide a "
    "1-10 suitability rating with a short justification grounded in repeated "
    "customer feedback? Reject any response that omits a populated 'Summary:' line "
    "or fails to state 'Suitability: <score>/10' followed by an evidence-based "
    "justification."
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


def build_user_prompt(persona: str, reviews: str) -> str:
    """Fill the config.USER_PROMPT template with persona and reviews text."""

    template = config.USER_PROMPT
    return template.replace("<persona>", persona).replace("<reviews>", reviews)


# ---------------------------------------------------------------------------
# SFT data generation
# ---------------------------------------------------------------------------
async def _synthesize_single_summary(
    sampler,
    record: PersonaRecord,
    include_reference: bool = True,
) -> dict:
    """Generate one conversation datum for SFT training."""
    system_prompt = config.SYSTEM_PROMPT.strip()
    user_prompt = build_user_prompt(record.persona, record.reviews).strip()

    if include_reference and record.reference_output:
        reference_block = "\n".join(reference.strip() for reference in record.reference_output if reference)
        user_prompt += f"{REFERENCE_PROMPT_SUFFIX}{reference_block}"

    # Lazily import heavy dependencies to keep RL-only runs lightweight.
    from scripts.test.utils import clear_content
    from tinker_cookbook import renderers

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

    from scripts.test.utils import TinkerSampler

    sampler = TinkerSampler(model_name=model_name, temperature=0.7, max_tokens=512, top_p=0.9)
    semaphore = asyncio.Semaphore(max(concurrency, 1))
    progress = tqdm(total=len(data), desc="[SFT] Generating", unit="prompt")

    async def bounded(idx: int, record: PersonaRecord) -> tuple[int, dict]:
        async with semaphore:
            synthesized = await _synthesize_single_summary(sampler, record)
        progress.update(1)
        return idx, synthesized

    try:
        tasks = [bounded(idx, record) for idx, record in enumerate(data)]
        collected: list[tuple[int, dict]] = await asyncio.gather(*tasks)
    finally:
        progress.close()

    collected.sort(key=lambda pair: pair[0])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for _, sample in collected:
            fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"[SFT] Wrote {len(collected)} examples to {output_path}")


# ---------------------------------------------------------------------------
# RL prompt generation
# ---------------------------------------------------------------------------
def build_prompt_conversation(record: PersonaRecord) -> list[dict]:
    messages: list[dict] = [
        {"role": "system", "content": config.SYSTEM_PROMPT.strip()},
        {"role": "user", "content": build_user_prompt(record.persona, record.reviews)},
    ]
    return messages


def generate_rl_dataset(
    data: Sequence[PersonaRecord],
    output_path: Path,
    rubric: str = DEFAULT_RUBRIC,
    seed: int | None = 42,
) -> None:
    """Write RL prompts to a single JSONL file."""
    if not data:
        raise ValueError("No persona records provided for RL synthesis.")

    if seed is not None:
        random.seed(seed)
    shuffled = list(data)
    random.shuffle(shuffled)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as train_f:
        for record in shuffled:
            payload = {
                "prompt_conversation": build_prompt_conversation(record),
                "reference": "\n".join(record.reference_output) or None,
                "rubric": rubric,
            }
            train_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(f"[RL] Wrote {len(shuffled)} prompts to {output_path}")

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
    parser.add_argument(
        "--rl-output",
        type=Path,
        default=DEFAULT_RL_OUTPUT,
        help="Output JSONL file for RL prompts (defaults to dataset/data/processed/rl/v1_rl_train.jsonl)",
    )
    parser.add_argument("--rubric", type=str, default=DEFAULT_RUBRIC)
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
        output_path = args.rl_output
        if output_path.suffix != ".jsonl":
            raise ValueError("--rl-output must point to a .jsonl file, e.g., dataset/data/processed/rl/v1_rl_train.jsonl")

        generate_rl_dataset(
            records,
            output_path=output_path,
            rubric=args.rubric,
            seed=args.seed,
        )
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
