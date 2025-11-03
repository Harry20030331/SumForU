# test.py
"""
Asynchronous testing script for Tinker-based models.
Supports both single prompt (direct) and batch JSON (concurrent) inference.
"""

import argparse
import asyncio
import config
from tinker_cookbook import renderers
from utils import build_model, build_prompt, update_config_from_args, pretty_model_output, TinkerSampler, print_conversation


async def run_inference(model_type: str):
    """
    Run asynchronous concurrent inference for multiple prompts loaded from a JSON file.
    Args:
        model_type (str): Model type, one of ['baseline', 'sft', 'dpo'].
    """
    system_prompt_info = "with" if config.USE_SYSTEM_PROMPT else "without"
    print("*****************************************************************")
    print(f"--- Running {model_type.upper()} model in {config.PROMPT_MODE.upper()} mode {system_prompt_info} system prompt ---")
    print("*****************************************************************\n")
    model: TinkerSampler = build_model(model_type)
    message_groups: list[list[renderers.Message]] = build_prompt()

    results = await asyncio.gather(*[model.generate(m) for m in message_groups])

    # Print results in order
    print("********************* Inference Results ***********************\n")
    for idx in range(len(message_groups)):
        print(f"----------  \n[Sample {idx + 1}]")
        print_conversation(message_groups[idx], results[idx])

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

    asyncio.run(run_inference(args.model_type))

if __name__ == "__main__":
    main()
