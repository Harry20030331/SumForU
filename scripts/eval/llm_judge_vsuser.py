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
# RM_MODEL_NAME_FOR_TOKENIZER = "openai/gpt-oss-120b"
RM_MODEL_NAME_FOR_TOKENIZER = "Qwen/Qwen3-235B-A22B-Instruct-2507"
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

def load_all_data(test_data_path: Path, output_paths: Dict[str, Path]) -> Tuple[List[Dict], Dict[str, List[str]]]:
    """Loads preprocessed test data and model outputs from all files."""
    with test_data_path.open("r", encoding="utf-8") as f:
        test_data = []
        for line in f:
            if line.strip():
                test_data.append(json.loads(line.strip()))
    print(f"Test data loaded successfully from {test_data_path}. Total samples: {len(test_data)}")

    model_outputs = {}
    for model_name, path in output_paths.items():
        if path and path.exists():
            with open(path, "r", encoding="utf-8") as f:
                model_outputs[model_name] = json.load(f)
            # Ensure model output list matches test data size for correspondence
            assert len(model_outputs[model_name]) == len(test_data), f"Mismatch for {model_name}"
        else:
            print(f"Output file for {model_name} not found at {path}. Skipping.")
            continue
    return test_data, model_outputs

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

async def run_full_evaluation(test_data: List[Dict], model_outputs: Dict[str, List[str]], methods: List[str]):
    """Iterates through all data points and all method pairs, running the three-dimensional evaluation and ranking."""
    
    # Initialize Tinker components (as per the provided template)
    tokenizer = get_tokenizer(RM_MODEL_NAME_FOR_TOKENIZER)
    renderer = renderers.get_renderer(RM_RENDERER_NAME, tokenizer=tokenizer)
    service_client = ServiceClient(base_url=BASE_URL)
    sampling_kwargs = {"base_model": RM_MODEL_NAME_FOR_TOKENIZER}
    preference_sampling_client = service_client.create_sampling_client(**sampling_kwargs)
    reward_model = PrometheusEvalPreferenceModelFromChatRenderer(
        renderer, preference_sampling_client, temperature=RM_TEMPERATURE
    )

    method_pairs = []
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            method_pairs.append((methods[i], methods[j]))
    
    # Store rankings per method: {method: {'rank_sum': 0, 'first_count': 0}}
    method_stats = defaultdict(lambda: {'rank_sum': 0, 'first_count': 0})
    
    print(f"Starting LLM Judge evaluation across {len(test_data)} test items and {len(method_pairs)} pairs...")
    
    for idx in tqdm(range(len(test_data)), desc="Processing cases"):
        item = test_data[idx]
        # Extract common context (prompt, reference)
        _, conversation, reference = extract_prompts_for_item(item)
        
        # Prepare model responses
        responses = {
            m: get_response_message(model_outputs[m][idx])
            for m in methods
        }
        
        # Initialize scores for this case
        case_scores = {m: 0 for m in methods}
        
        # Compare all pairs in all dimensions
        tasks = []
        for m_A, m_B in method_pairs:
            for dim, rubric in JUDGE_RUBRICS.items():
                task = evaluate_pair_dimension(
                    reward_model,
                    conversation,
                    responses[m_A],
                    responses[m_B],
                    rubric,
                    reference
                )
                tasks.append((m_A, m_B, task))
        
        # Run tasks for this case
        results = await asyncio.gather(*[t[2] for t in tasks])
        
        for (m_A, m_B, _), score in zip(tasks, results):
            case_scores[m_A] += score
            case_scores[m_B] += (1 - score)
        
        # Sort methods by score descending (higher score better)
        sorted_methods = sorted(case_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Assign ranks (1-based)
        for rank, (method, score) in enumerate(sorted_methods, 1):
            method_stats[method]['rank_sum'] += rank
            if rank == 1:
                method_stats[method]['first_count'] += 1
    
    print("\n" + "="*80)
    print("LLM JUDGE RANKING REPORT")
    print(f"Total Test Items: {len(test_data)}")
    print("="*80)
    
    for method in methods:
        stats = method_stats[method]
        avg_rank = stats['rank_sum'] / len(test_data)
        first_rate = stats['first_count'] / len(test_data)
        print(f"\n--- METHOD: {method.upper()} ---")
        print(f"Average Rank: {avg_rank:.3f}")
        print(f"First Place Rate: {first_rate:.3f}")


if __name__ == "__main__":
    # Add project root to path if not already done (copied from user template)
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    
    parser = argparse.ArgumentParser(description="Run LLM Judge evaluation for multiple models.")
    parser.add_argument(
        "--test-data-path",
        type=Path,
        default=Path("results/user_study/user_study.jsonl"),
        help="Path to test data JSONL file",
    )
    parser.add_argument(
        "--baseline-path",
        type=Path,
        default=Path("results/user_study/baseline_answers.json"),
        help="Path to baseline model outputs JSON",
    )
    parser.add_argument(
        "--sft-path",
        type=Path,
        default=Path("results/user_study/sft_answers.json"),
        help="Path to SFT model outputs JSON",
    )
    parser.add_argument(
        "--pe-path",
        type=Path,
        default=Path("results/user_study/pe_answers.json"),
        help="Path to PE model outputs JSON",
    )
    parser.add_argument(
        "--rl-path",
        type=Path,
        default=Path("results/user_study/rl_answers.json"),
        help="Path to RL model outputs JSON",
    )
    args = parser.parse_args()

    output_paths = {
        "baseline": args.baseline_path,
        "sft": args.sft_path,
        "pe": args.pe_path,
        "rl": args.rl_path,
    }
    
    # NOTE: Execution requires the user environment to have the 'tinker', 'tinker_cookbook', 
    # and 'scripts.train.prometheus_types' dependencies installed and configured to access the 235B model.
    test_data, model_outputs = load_all_data(args.test_data_path, output_paths)
    
    methods = [m for m in ["baseline", "pe", "sft", "rl"] if m in model_outputs]
    if not methods:
        print("Error: At least one model output path must be provided.")
        sys.exit(1)
    
    asyncio.run(run_full_evaluation(test_data, model_outputs, methods))