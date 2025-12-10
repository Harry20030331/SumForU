import json
import os
import sys
import asyncio
import argparse
from collections import defaultdict
from typing import List, Dict, Any, Tuple
from tqdm.asyncio import tqdm
from tinker_cookbook import renderers, model_info
from tinker import ServiceClient, types
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from scripts.train.prometheus_types import (
    PrometheusEvalComparison,
    PrometheusEvalPreferenceModelFromChatRenderer,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from scripts.test.utils import build_user_prompt
from scripts.test import config

# NOTE: Dependencies from the provided template are assumed to be available
# from tinker_cookbook import renderers, model_info
# from tinker import ServiceClient, types
# from scripts.train.prometheus_types import PrometheusEvalComparison, PrometheusEvalPreferenceModelFromChatRenderer
# from tinker_cookbook.tokenizer_utils import get_tokenizer
# from scripts.test.utils import build_user_prompt
# from scripts.test import config

# --- 1. CONFIGURATION AND INITIAL SETUP (UNCHANGED) ---

BASE_URL = None
RM_MODEL_NAME_FOR_TOKENIZER = "openai/gpt-oss-120b"
# RM_MODEL_NAME_FOR_TOKENIZER = "Qwen/Qwen3-235B-A22B-Instruct-2507"
# RM_MODEL_NAME_FOR_TOKENIZER = "meta-llama/Llama-3.3-70B-Instruct"
RM_RENDERER_NAME = model_info.get_recommended_renderer_name(RM_MODEL_NAME_FOR_TOKENIZER)
RM_MODEL_PATH = None
RM_TEMPERATURE = 0.0

# Define file paths (assuming they are relative to the script location)
# TEST_DATA_PATH = "v1_test_preprocessed.json"
# OUTPUT_PATHS = {
#     "baseline": "results/v1_test_baseline.json",
#     "sft": "results/v1_test_sft.json",
#     "pe": "results/v1_test_pe.json",
#     "rl": "results/v1_test_rl.json",
# }

# --- 2. LLM JUDGE RUBRIC DEFINITIONS ---

# Dimension 1: Summary-Suitability Consistency (Adapter)
RUBRIC_CONSISTENCY = (
    "Compare Completion A and B. Which pair demonstrates stronger **internal consistency** between "
    "the qualitative Summary and the quantitative Suitability Score? A high score (8-10) must be "
    "supported by overwhelmingly positive synthesis and strong utility claims in the Summary. A low score (1-3) "
    "must be justified by clearly articulated and factually grounded drawbacks. Penalize responses where the "
    "Summary's sentiment contradicts the Score's magnitude (e.g., highly negative summary with 9/10 score). "
    "You must choose either A or B as better; there are no ties."
)

# Dimension 2: Factual Grounding (Hallucination Detection)
RUBRIC_GROUNDING = (
    "Compare Completion A and B. Which summary is **more factually grounded** and contains **fewer hallucinations "
    "or unverified claims** relative to the original Review Set? A summary should only contain claims that can be "
    "logically inferred or directly supported by the evidence in the Review Set. If both summaries contain minor "
    "abstractions, select the one whose claims are more robustly traceable to the source documents. "
    "Factual consistency is a hard constraint for selection. "
    "You must choose either A or B as better; there are no ties."
)

# Dimension 3: Persona Alignment (Persona Response)
RUBRIC_PERSONA = (
    "Compare Completion A and B. Which summary **better reflects the user's explicit Persona priorities**? The preferred "
    "summary must prioritize the discussion of aspects most critical to the Persona (e.g., emphasizing 'durability' over "
    "'price' if instructed), and synthesize evidence accordingly. It must present the evidence in a tone that is sensitive "
    "to the Persona's values. Select the summary that offers the most actionable and relevant personalized insight "
    "for the user's decision-making. "
    "You must choose either A or B as better; there are no ties."
)

JUDGE_RUBRICS = {
    "Consistency": RUBRIC_CONSISTENCY,
    "Grounding": RUBRIC_GROUNDING,
    "Persona": RUBRIC_PERSONA,
}

# --- 3. DATA LOADING AND PREPARATION ---

def extract_prompts_for_item(item: Dict) -> Tuple[str, List[Dict], str]:
    """Builds the user prompt and conversation structure for a single item."""
    # Assuming config.SYSTEM_PROMPT and build_user_prompt are available from imports
    input_text = build_user_prompt(item["persona"], item["reviews"])
    prompt_conversation = [{"role": "user", "content": input_text}]  # List of dicts
    reference = "\n".join(item.get("reference_output", [])) or None
    return input_text, prompt_conversation, reference

def get_response_message(content: str) -> List:
    """Wraps model output content into the required renderers.Message format."""
    return [{"role": "assistant", "content": content}]

# --- 4. CORE EVALUATION LOGIC ---

async def evaluate_pair_dimension(
    reward_model, 
    conversation: List, 
    response_A: List, 
    response_B: List, 
    rubric: str, 
    reference: str
) -> int:
    """Runs a single asynchronous comparison for one dimension."""
    comparison = PrometheusEvalComparison(
        prompt_conversation=conversation,
        completion_A=response_A,
        completion_B=response_B,
        rubric=rubric,
        reference=reference,
    )
    # The reward model returns a score: < 0 if A is better, > 0 if B is better (based on the if "A" return -1, "B" return 1).
    score = await reward_model(comparison)
    # Convert to 1 if A better, 0 if B better.
    return 1 if score < 0 else 0

async def run_full_evaluation(test_data: List[Dict], model_outputs: Dict[str, List[str]], methods: List[str], category: str, output_dir: Path):
    """Iterates through all data points and all method pairs, running the three-dimensional evaluation."""
    
    # Initialize Tinker components (as per the provided template)
    tokenizer = get_tokenizer(RM_MODEL_NAME_FOR_TOKENIZER)
    renderer = renderers.get_renderer(RM_RENDERER_NAME, tokenizer=tokenizer)
    service_client = ServiceClient(base_url=BASE_URL)
    sampling_kwargs = {"base_model": RM_MODEL_NAME_FOR_TOKENIZER}
    preference_sampling_client = service_client.create_sampling_client(**sampling_kwargs)
    reward_model = PrometheusEvalPreferenceModelFromChatRenderer(
        renderer, preference_sampling_client, temperature=RM_TEMPERATURE
    )

    # methods = ["baseline", "pe", "sft", "rl"]
    method_pairs = []
    
    # Define all unique pairs (e.g., baseline vs pe, pe vs sft, etc.)
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            method_pairs.append((methods[i], methods[j]))
    
    # Store aggregated results per method: {method: {'Consistency': total_score, 'Grounding': total_score, ...}}
    method_scores = defaultdict(lambda: defaultdict(int))
    
    print(f"Starting LLM Judge evaluation for category {category} across {len(test_data)} test items and {len(method_pairs)} pairs...")
    
    # Collect all tasks across all items
    all_tasks = []
    global_task_map = {}  # key: global_index, value: (item_idx, m_A, m_B, dim)
    task_to_global_index = {}  # key: task, value: global_index
    
    for idx, item in enumerate(test_data):
        # Extract common context (prompt, reference)
        _, conversation, reference = extract_prompts_for_item(item)
        
        # Prepare model responses
        responses = {
            m: get_response_message(model_outputs[m][idx])
            for m in methods
        }
        
        for m_A, m_B in method_pairs:
            for dim, rubric in JUDGE_RUBRICS.items():
                coro = evaluate_pair_dimension(
                    reward_model,
                    conversation,
                    responses[m_A],
                    responses[m_B],
                    rubric,
                    reference
                )
                task = asyncio.create_task(coro)
                global_index = len(all_tasks)
                all_tasks.append(task)
                task_to_global_index[task] = global_index
                global_task_map[global_index] = (idx, m_A, m_B, dim)
    
    # Run all comparisons in parallel with real-time progress
    results = await tqdm.gather(*all_tasks, desc="Processing all comparisons")
    
    # Aggregate results per method
    for global_index, score in enumerate(results):
        item_idx, m_A, m_B, dim = global_task_map[global_index]
        # Score: 1 if A better, 0 if B better.
        method_scores[m_A][dim] += score
        method_scores[m_B][dim] += (1 - score)
    
    print("\n" + "="*80)
    print(f"LLM JUDGE (Qwen3-235B) THREE-DIMENSIONAL COMPARISON REPORT FOR {category.upper()}")
    print(f"Total Test Items: {len(test_data)}")
    print(f"Scoring: Each method's score = (wins against other methods) / {(len(methods) - 1) * len(test_data)}. Overall = average of three dimensions.")
    print("="*80)
    
    results_dict = {
        "category": category,
        "methods": {},
    }
    
    for method in methods:
        dim_scores = method_scores[method]
        total_score = sum(dim_scores.values())
        overall_avg = total_score / len(JUDGE_RUBRICS) / ((len(methods) - 1) * len(test_data)) if JUDGE_RUBRICS else 0
        
        print(f"\n--- METHOD: {method.upper()} ---")
        print(f"Overall Average Score: {overall_avg:.3f}")
        print(f"Total Wins: {total_score} / {(len(methods) - 1) * len(test_data) * len(JUDGE_RUBRICS)}")
        print(f"| {'Dimension':<15} | Win Rate |")
        print("|" + "-"*15 + "|" + "-"*9 + "|")
        dim_dict = {}
        for dim in JUDGE_RUBRICS.keys():
            score = dim_scores[dim]
            win_rate = score / ((len(methods) - 1) * len(test_data))
            print(f"| {dim:<15} | {win_rate:.3f} |")
            dim_dict[dim] = win_rate
        results_dict["methods"][method] = {
            "overall_avg": overall_avg,
            "dimensions": dim_dict,
        }
    
    # Save to JSON
    output_path = output_dir / "llm_metrics.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    # Add project root to path if not already done (copied from user template)
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    
    parser = argparse.ArgumentParser(description="Run LLM Judge evaluation for generalization tasks.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Input directory containing generalization results, e.g., results/generalization/",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        required=True,
        help="Ground truth directory containing preprocessed generalization data, e.g., dataset/data/generalization/preprocessed/",
    )
    args = parser.parse_args()

    # Get categories from gt-dir
    gt_files = list(args.gt_dir.glob("preprocessed_*.json"))
    categories = [f.stem.replace("preprocessed_", "") for f in gt_files]

    for category in categories:
        print(f"\nProcessing category: {category}")
        gt_path = args.gt_dir / f"preprocessed_{category}.json"
        with gt_path.open("r", encoding="utf-8") as f:
            test_data = json.load(f)

        category_dir = args.input_dir / category
        output_paths = {
            "baseline": category_dir / "baseline.json",
            "sft": category_dir / "sft.json",
            "pe": category_dir / "pe.json",
            "rl": category_dir / "rl.json",
        }
        
        model_outputs = {}
        for model_name, path in output_paths.items():
            if path.exists():
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        model_outputs[model_name] = json.load(f)
                    assert len(model_outputs[model_name]) == len(test_data)
                except (FileNotFoundError, AssertionError):
                    print(f"Error loading or validating output for {model_name} at {path}. Skipping.")
                    continue
        
        methods = list(model_outputs.keys())
        if not methods:
            print(f"No valid model outputs for category {category}. Skipping.")
            continue
        
        asyncio.run(run_full_evaluation(test_data, model_outputs, methods, category, category_dir))