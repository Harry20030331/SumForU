# test.py
"""
Asynchronous testing script for Tinker-based models.
Supports both single prompt (direct) and batch JSON (concurrent) inference.
"""

import argparse
import asyncio
import json
import tempfile
from pathlib import Path

from tinker_cookbook import renderers

from . import config
from .utils import (
    TinkerSampler,
    build_model,
    build_prompt,
    clear_content,
    print_conversation,
    update_config_from_args,
)


async def run_inference(model_type: str, target_file: str = None, input_data=None):
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
    message_groups: list[list[renderers.Message]] = build_prompt(input_data)

    results = await asyncio.gather(*[model.generate(m) for m in message_groups])
    
    debug_mode = config.DEBUG_MODE if config.PROMPT_MODE == config.JSON else True
    if debug_mode:
        # load reference outputs if in JSON mode
        if config.PROMPT_MODE == config.JSON:
            references = [item.get("reference_output", []) for item in input_data]

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
    # Get available categories from sft/test directory
    sft_test_dir = Path("dataset/data/processed/sft/test")
    sft_test_files = sorted(sft_test_dir.glob("*.jsonl"))
    categories = sorted([f.stem for f in sft_test_files])
    
    parser = argparse.ArgumentParser(description="Run async model test under Tinker framework.")
    parser.add_argument("--task", type=str, default="normal", choices=["normal", "generalization"], help="Task type: normal or generalization.")
    parser.add_argument("--model_type", type=str, default="all", choices=[config.BASELINE, config.SFT, config.RL, config.PE, "all"])
    parser.add_argument("--output", type=str, default=None, help="Path to the output file or directory. If model_type is 'all', this should be a directory.")
    parser.add_argument("--user_input", type=str, default="v1_test_preprocessed.json", help="Path to JSON file or single prompt text.")
    parser.add_argument("--category", type=str, default=None, help=f"Category to test: {', '.join(categories)} or 'whole_dataset' to merge all.")

    parser.add_argument("--temperature", type=float, help="Override temperature.")
    parser.add_argument("--max_tokens", type=int, help="Override max tokens.")
    parser.add_argument("--use_system_prompt", type=str, help="Use system prompt (True/False).")
    parser.add_argument("--system_prompt", type=str, help="Override system prompt text.")
    parser.add_argument("--debug_mode", type=str, help="Enable debug mode (True/False).")

    args = parser.parse_args()

    if args.task == "generalization":
        # Handle generalization task
        preprocessed_dir = Path("dataset/data/generalization/preprocessed")
        gen_categories = ["Arts_Crafts_and_Sewing", "Industrial_and_Scientific", "Video_Games"]
        config.PROMPT_MODE = config.JSON
        update_config_from_args(args)
        for category in gen_categories:
            input_file = preprocessed_dir / f"preprocessed_{category}.json"
            with open(input_file, 'r', encoding='utf-8') as f:
                input_data = json.load(f)
            output_dir = Path("results/generalization") / category
            output_dir.mkdir(parents=True, exist_ok=True)
            model_types = [config.BASELINE, config.SFT, config.RL, config.PE]
            for model_type in model_types:
                print(f"\n{'='*20} Running {model_type.upper()} for {category} {'='*20}\n")
                if model_type == config.BASELINE:
                    config.USE_SYSTEM_PROMPT = False
                else:
                    config.USE_SYSTEM_PROMPT = True
                target_file = output_dir / f"{model_type}.json"
                asyncio.run(run_inference(model_type, str(target_file), input_data))
        return

    # Original logic for normal task
    # Set category and PROMPT_PATH first
    if args.category == "whole_dataset":
        # Merge all sft/test files into one dataset
        merged_data = []
        for f in sft_test_files:
            with open(f, 'r', encoding='utf-8') as file:
                for line in file:
                    merged_data.append(json.loads(line.strip()))
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(merged_data, temp_file, indent=2, ensure_ascii=False)
        temp_file.close()
        config.PROMPT_PATH = temp_file.name
        input_data = merged_data
        print("-----------------------------------------------------------------")
        print(f"Whole dataset merge completed: Merged {len(sft_test_files)} files into {len(merged_data)} items")
        print("-----------------------------------------------------------------")
    elif args.category in categories:
        # Load specific category
        category_file = sft_test_dir / f"{args.category}.jsonl"
        config.PROMPT_PATH = str(category_file)
        with open(category_file, 'r', encoding='utf-8') as file:
            input_data = [json.loads(line.strip()) for line in file if line.strip()]
        print("-----------------------------------------------------------------")
        print(f"Using category: {args.category}")
        print("-----------------------------------------------------------------")
    else:
        # Default to whole_dataset if no category specified
        merged_data = []
        for f in sft_test_files:
            with open(f, 'r', encoding='utf-8') as file:
                for line in file:
                    merged_data.append(json.loads(line.strip()))
        input_data = merged_data

    if args.output is None and config.PROMPT_MODE == config.JSON:
        raise ValueError("Output must be specified in JSON mode to save results.")
    
    # Update config based on command-line args
    update_config_from_args(args)

    if args.model_type == "all":
        if args.output is None:
            raise ValueError("Output directory must be specified when model_type is 'all'.")
        target_dir = Path(args.output)
        target_dir.mkdir(parents=True, exist_ok=True)
        model_types = [config.BASELINE, config.SFT, config.RL, config.PE]
        for model_type in model_types:
            print(f"\n{'='*20} Running {model_type.upper()} {'='*20}\n")
            # Set USE_SYSTEM_PROMPT for this model_type
            if model_type == config.BASELINE:
                config.USE_SYSTEM_PROMPT = False
            else:
                config.USE_SYSTEM_PROMPT = True
            target_file = target_dir / f"{model_type}.json"
            asyncio.run(run_inference(model_type, str(target_file), input_data))
    else:
        # Set USE_SYSTEM_PROMPT based on model_type
        if args.model_type == config.BASELINE:
            config.USE_SYSTEM_PROMPT = False
        else:  # pe, sft, rl
            config.USE_SYSTEM_PROMPT = True
        asyncio.run(run_inference(args.model_type, args.output, input_data))

if __name__ == "__main__":
    main()
