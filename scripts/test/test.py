# test.py
"""
Asynchronous testing script for Tinker-based models.
Supports both single prompt (direct) and batch JSON (concurrent) inference.
"""

import argparse
import asyncio
import config
from tinker_cookbook import renderers
from utils import build_model, build_prompt, update_config_from_args, pretty_model_output, TinkerSampler


# ---------- Single Prompt Inference (Direct Mode) ----------
async def run_inference_direct(model_type: str):
    """
    Run asynchronous inference for a single prompt.
    Args:
        model_type (str): Model type, one of ['baseline', 'sft', 'dpo'].
        prompt_input (str): Optional single prompt string.
    """
    system_prompt_info = "with" if config.USE_SYSTEM_PROMPT else "without"
    print("*****************************************************************")
    print(f"--- Running {model_type.upper()} model in DIRECT mode {system_prompt_info} system prompt ---")
    print("*****************************************************************\n")
    model: TinkerSampler = build_model(model_type)
    messages: list[renderers.Message] = build_prompt()

    response = await model.generate(messages)
    pretty_model_output(response["content"])


# ---------- Batch Inference (JSON Mode) ----------
async def run_inference_json(model_type: str):
    """
    Run asynchronous concurrent inference for multiple prompts loaded from a JSON file.
    Args:
        model_type (str): Model type, one of ['baseline', 'sft', 'dpo'].
    """
    system_prompt_info = "with" if config.USE_SYSTEM_PROMPT else "without"
    print(f"--- Running {model_type.upper()} model in JSON mode (async batch) {system_prompt_info} system prompt ---\n")
    model: TinkerSampler = build_model(model_type)
    message_groups: list[list[renderers.Message]] = build_prompt()
    
    print(f"Loaded {len(message_groups)} samples.\n")

    async def get_output(msgs, idx):
        response = await model.generate(msgs)
        return idx, response["content"]

    # Run inference concurrently using asyncio.gather
    tasks = [get_output(m, i) for i, m in enumerate(message_groups)]
    results = await asyncio.gather(*tasks)

    # Print results in order
    for idx, output in sorted(results, key=lambda x: x[0]):
        print(f"\n[Sample {idx + 1}] -------------------")
        pretty_model_output(output)

# ---------- Entry Point ----------
def main():
    parser = argparse.ArgumentParser(description="Run async model test under Tinker framework.")
    parser.add_argument("--model_type", type=str, default=config.BASELINE, choices=[config.BASELINE, config.SFT, config.DPO])
    parser.add_argument("--active_mode", type=str, choices=[config.JSON, config.DIRECT])
    parser.add_argument("--user_input", help="Path to JSON file or single prompt text.")

    parser.add_argument("--temperature", type=float, help="Override temperature.")
    parser.add_argument("--max_tokens", type=int, help="Override max tokens.")
    parser.add_argument("--use_system_prompt", type=str, help="Use system prompt (True/False).")
    parser.add_argument("--system_prompt", type=str, help="Override system prompt text.")

    args = parser.parse_args()

    # Update config based on command-line args
    update_config_from_args(args)

    if args.active_mode == config.JSON:
        asyncio.run(run_inference_json(args.model_type))
    else:
        asyncio.run(run_inference_direct(args.model_type))

if __name__ == "__main__":
    main()
