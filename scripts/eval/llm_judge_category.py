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

# --- 1. CONFIGURATION AND INITIAL SETUP (UNCHANGED) ---

BASE_URL = None
RM_MODEL_NAME_FOR_TOKENIZER = "openai/gpt-oss-120b"
# RM_MODEL_NAME_FOR_TOKENIZER = "Qwen/Qwen3-235B-A22B-Instruct-2507"
# RM_MODEL_NAME_FOR_TOKENIZER = "meta-llama/Llama-3.3-70B-Instruct"
RM_RENDERER_NAME = model_info.get_recommended_renderer_name(RM_MODEL_NAME_FOR_TOKENIZER)
RM_MODEL_PATH = None
RM_TEMPERATURE = 0.0

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

def load_grouped_data(test_data_path: Path, output_paths: Dict[str, Path]) -> Tuple[Dict[str, List[Dict]], Dict[str, Dict[str, List[str]]]]:
    """Loads and groups data by category."""
    if test_data_path.is_dir():
        test_data = []
        for jsonl_file in sorted(test_data_path.glob("*.jsonl")):
            category = jsonl_file.stem  # Use filename as category
            with jsonl_file.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line.strip())
                        test_data.append((category, item))
        print(f"Test data loaded successfully from directory. Total samples: {len(test_data)}")
    else:
        with test_data_path.open("r", encoding="utf-8") as f:
            test_list = json.load(f)
        test_data = []
        for item in test_list:
            category = item.get("category", "unknown")
            test_data.append((category, item))
        print(f"Test data loaded successfully from file. Total samples: {len(test_data)}")

    # Group test_data by category
    test_data_grouped = {}
    for category, item in test_data:
        if category not in test_data_grouped:
            test_data_grouped[category] = []
        test_data_grouped[category].append(item)

    # Load model outputs and group by category
    model_outputs_grouped = {}
    for model_name, path in output_paths.items():
        if path:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    outputs = json.load(f)
                # Assume outputs are in the same order as test_data
                idx = 0
                for category, items in test_data_grouped.items():
                    if category not in model_outputs_grouped:
                        model_outputs_grouped[category] = {}
                    model_outputs_grouped[category][model_name] = outputs[idx:idx + len(items)]
                    idx += len(items)
                assert idx == len(outputs)
            except (FileNotFoundError, AssertionError):
                print(f"Error loading or validating output for {model_name} at {path}. Skipping.")
                continue
    return test_data_grouped, model_outputs_grouped

def extract_prompts_for_item(item: Dict) -> Tuple[str, List[Dict], str]:
    """Builds the user prompt and conversation structure for a single item."""
    input_text = build_user_prompt(item["persona"], item["reviews"])
    prompt_conversation = [{"role": "user", "content": input_text}]
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
    score = await reward_model(comparison)
    return 1 if score < 0 else 0

async def run_full_evaluation_for_category(reward_model, test_data: List[Dict], model_outputs: Dict[str, List[str]], methods: List[str], category: str):
    """Iterates through all data points and all method pairs, running the three-dimensional evaluation for a category."""
    
    method_pairs = []
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            method_pairs.append((methods[i], methods[j]))
    
    method_scores = defaultdict(lambda: defaultdict(int))
    
    print(f"Starting LLM Judge evaluation for category {category} across {len(test_data)} test items and {len(method_pairs)} pairs...")
    
    all_tasks = []
    global_task_map = {}
    
    for idx, item in enumerate(test_data):
        _, conversation, reference = extract_prompts_for_item(item)
        
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
                global_task_map[global_index] = (idx, m_A, m_B, dim)
    
    results = await tqdm.gather(*all_tasks, desc=f"Processing {category}")
    
    for global_index, score in enumerate(results):
        item_idx, m_A, m_B, dim = global_task_map[global_index]
        method_scores[m_A][dim] += score
        method_scores[m_B][dim] += (1 - score)
    
    print(f"\n" + "="*80)
    print(f"LLM JUDGE (Qwen3-235B) THREE-DIMENSIONAL COMPARISON REPORT FOR {category.upper()}")
    print(f"Total Test Items: {len(test_data)}")
    print(f"Scoring: Each method's score = (wins against other methods) / {(len(methods) - 1) * len(test_data)}. Overall = average of three dimensions.")
    print("="*80)
    
    for method in methods:
        dim_scores = method_scores[method]
        total_score = sum(dim_scores.values())
        overall_avg = total_score / len(JUDGE_RUBRICS) / ((len(methods) - 1) * len(test_data)) if JUDGE_RUBRICS else 0
        
        print(f"\n--- METHOD: {method.upper()} ---")
        print(f"Overall Average Score: {overall_avg:.3f}")
        print(f"Total Wins: {total_score} / {(len(methods) - 1) * len(test_data) * len(JUDGE_RUBRICS)}")
        print(f"| {'Dimension':<15} | Win Rate |")
        print("|" + "-"*15 + "|" + "-"*9 + "|")
        for dim in JUDGE_RUBRICS.keys():
            score = dim_scores[dim]
            win_rate = score / ((len(methods) - 1) * len(test_data))
            print(f"| {dim:<15} | {win_rate:.3f} |")
    
    return method_scores

async def run_full_evaluation(test_data_grouped: Dict[str, List[Dict]], model_outputs_grouped: Dict[str, Dict[str, List[str]]], methods: List[str]):
    # Initialize Tinker components once
    tokenizer = get_tokenizer(RM_MODEL_NAME_FOR_TOKENIZER)
    renderer = renderers.get_renderer(RM_RENDERER_NAME, tokenizer=tokenizer)
    service_client = ServiceClient(base_url=BASE_URL)
    sampling_kwargs = {"base_model": RM_MODEL_NAME_FOR_TOKENIZER}
    preference_sampling_client = service_client.create_sampling_client(**sampling_kwargs)
    reward_model = PrometheusEvalPreferenceModelFromChatRenderer(
        renderer, preference_sampling_client, temperature=RM_TEMPERATURE
    )

    results = {}
    for category, test_data in test_data_grouped.items():
        model_outputs = model_outputs_grouped.get(category, {})
        category_methods = [m for m in methods if m in model_outputs]
        if not category_methods:
            continue
        method_scores = await run_full_evaluation_for_category(reward_model, test_data, model_outputs, category_methods, category)
        results[category] = method_scores
    
    # Compute overall by aggregating category scores
    print("\nComputing overall...")
    overall_scores = defaultdict(lambda: defaultdict(int))
    total_test_items = 0
    overall_methods = set()
    for category, method_scores in results.items():
        if category == "overall":
            continue
        category_test_items = len(test_data_grouped[category])
        total_test_items += category_test_items
        for method, dim_scores in method_scores.items():
            overall_methods.add(method)
            for dim, score in dim_scores.items():
                overall_scores[method][dim] += score
    
    overall_methods = list(overall_methods)
    if overall_methods:
        # Print overall report
        print(f"\n" + "="*80)
        print(f"LLM JUDGE (Qwen3-235B) THREE-DIMENSIONAL COMPARISON REPORT FOR OVERALL")
        print(f"Total Test Items: {total_test_items}")
        print(f"Scoring: Each method's score = (wins against other methods) / {(len(overall_methods) - 1) * total_test_items}. Overall = average of three dimensions.")
        print("="*80)
        
        for method in overall_methods:
            dim_scores = overall_scores[method]
            total_score = sum(dim_scores.values())
            overall_avg = total_score / len(JUDGE_RUBRICS) / ((len(overall_methods) - 1) * total_test_items) if JUDGE_RUBRICS else 0
            
            print(f"\n--- METHOD: {method.upper()} ---")
            print(f"Overall Average Score: {overall_avg:.3f}")
            print(f"Total Wins: {total_score} / {(len(overall_methods) - 1) * total_test_items * len(JUDGE_RUBRICS)}")
            print(f"| {'Dimension':<15} | Win Rate |")
            print("|" + "-"*15 + "|" + "-"*9 + "|")
            for dim in JUDGE_RUBRICS.keys():
                score = dim_scores[dim]
                win_rate = score / ((len(overall_methods) - 1) * total_test_items)
                print(f"| {dim:<15} | {win_rate:.3f} |")
        
        results["overall"] = overall_scores
    
    # Write to JSON
    with open("llm_metric_category.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Results written to llm_metric_category.json")

if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    
    parser = argparse.ArgumentParser(description="Run LLM Judge evaluation for multiple models, grouped by category.")
    parser.add_argument(
        "--test-data-path",
        type=Path,
        required=True,
        help="Path to test data: either a JSON file or a directory containing .jsonl files to merge",
    )
    parser.add_argument(
        "--baseline-path",
        type=Path,
        required=False,
        default=None,
        help="Path to baseline model outputs JSON",
    )
    parser.add_argument(
        "--sft-path",
        type=Path,
        required=False,
        default=None,
        help="Path to SFT model outputs JSON",
    )
    parser.add_argument(
        "--pe-path",
        type=Path,
        required=False,
        default=None,
        help="Path to PE model outputs JSON",
    )
    parser.add_argument(
        "--rl-path",
        type=Path,
        required=False,
        default=None,
        help="Path to RL model outputs JSON",
    )
    args = parser.parse_args()

    output_paths = {
        "baseline": args.baseline_path,
        "sft": args.sft_path,
        "pe": args.pe_path,
        "rl": args.rl_path,
    }
    
    test_data_grouped, model_outputs_grouped = load_grouped_data(args.test_data_path, output_paths)
    
    methods = [m for m in ["baseline", "pe", "sft", "rl"] if output_paths[m] is not None]
    if not methods:
        print("Error: At least one model output path must be provided.")
        sys.exit(1)
    
    asyncio.run(run_full_evaluation(test_data_grouped, model_outputs_grouped, methods))