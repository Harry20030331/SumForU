# test.py
"""
Asynchronous testing script for Tinker-based models.
Supports both single prompt (direct) and batch JSON (concurrent) inference.
"""

import argparse
import asyncio
import config
import json
from tinker_cookbook import renderers
from utils import build_model, build_prompt, update_config_from_args, clear_content, TinkerSampler, print_conversation


async def run_inference(model_type: str, target_file: str = None):
    """
    Run asynchronous concurrent inference for multiple prompts loaded from a JSON file.
    Args:
        model_type (str): Model type, one of ['baseline', 'sft', 'rl'].
    """
    system_prompt_info = "with" if config.USE_SYSTEM_PROMPT else "without"
    print("*****************************************************************")
    print(f"--- Running {model_type.upper()} model in {config.PROMPT_MODE.upper()} mode {system_prompt_info} system prompt ---")
    print("*****************************************************************\n")
    model: TinkerSampler = build_model(model_type)
    message_groups: list[list[renderers.Message]] = build_prompt()

    results = await asyncio.gather(*[model.generate(m) for m in message_groups])
    
    debug_mode = config.DEBUG_MODE if config.PROMPT_MODE == config.JSON else True
    if debug_mode:
        # load reference ouputs if in JSON mode
        if config.PROMPT_MODE == config.JSON:
            with open(config.PROMPT_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            references = [item.get("reference_output", []) for item in data]

        # Print results in order
        print("********************* Inference Results ***********************\n")
        for idx in range(len(message_groups)):
            print(f"----------  \n[Sample {idx + 1}]")
            print_conversation(message_groups[idx], results[idx], references[idx])
    
    if config.PROMPT_MODE == config.JSON and target_file:
        # Save results to target file
        with open(target_file, "w", encoding="utf-8") as f:
            # only save generated outputs of content messages, exclude roles
            json.dump([r["content"] for r in results], f, indent=4)
        print("**************** Results saved to", target_file, "****************")


# ---------- Entry Point ----------
def main():
    parser = argparse.ArgumentParser(description="Run async model test under Tinker framework.")
    parser.add_argument("--model_type", type=str, default=config.BASELINE, choices=[config.BASELINE, config.SFT, config.RL])
    parser.add_argument("--target_file", type=str, default=None, help="Path to the target JSON file for saving results.")
    parser.add_argument("--active_mode", type=str, choices=[config.JSON, config.DIRECT])
    parser.add_argument("--user_input", help="Path to JSON file or single prompt text.")

    parser.add_argument("--temperature", type=float, help="Override temperature.")
    parser.add_argument("--max_tokens", type=int, help="Override max tokens.")
    parser.add_argument("--use_system_prompt", type=str, help="Use system prompt (True/False).")
    parser.add_argument("--system_prompt", type=str, help="Override system prompt text.")
    parser.add_argument("--debug_mode", type=str, help="Enable debug mode (True/False).")

    args = parser.parse_args()

    if args.target_file is None and config.PROMPT_MODE == config.JSON:
        raise ValueError("Target file must be specified in JSON mode to save results.")
    
    # Update config based on command-line args
    update_config_from_args(args)

    asyncio.run(run_inference(args.model_type, args.target_file))

if __name__ == "__main__":
    main()
