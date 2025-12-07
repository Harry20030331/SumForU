"""
Independent script to call Reward Model to judge which of two responses is better.
Dependencies: tinker, tinker_cookbook, scripts.train.prometheus_types
"""
import asyncio
import json
from tinker_cookbook import renderers, model_info
from tinker import ServiceClient, types
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from scripts.train.prometheus_types import (
    PrometheusEvalComparison,
    PrometheusEvalPreferenceModelFromChatRenderer,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from scripts.test.utils import build_user_prompt
from scripts.test import config

# --------- Configuration Section (Consistent with RL Process) ---------
BASE_URL = None  # Reward model service address, None means use default
RM_MODEL_NAME_FOR_TOKENIZER = "Qwen/Qwen3-235B-A22B-Instruct-2507"  # Reward model name
RM_RENDERER_NAME = model_info.get_recommended_renderer_name(RM_MODEL_NAME_FOR_TOKENIZER)  # Reward model renderer name
RM_MODEL_PATH = None  # Reward model local path, None means use base_model
RM_TEMPERATURE = 0.0  # Reward model temperature

# --------- Input Section (Load from test data) ---------
# Load the first case from dataset/data/raw/v1_test_preprocessed.json
with open("dataset/data/raw/v1_test_preprocessed.json", "r", encoding="utf-8") as f:
    data = json.load(f)
first_item = data[0]
input_text = build_user_prompt(first_item["persona"], first_item["reviews"])
prompt_conversation = [
    {"role": "system", "content": config.SYSTEM_PROMPT.strip()},
    {"role": "user", "content": input_text},
]
rubric = (
    "Does the assistant deliver a concise 2-3 sentence summary that focuses on "
    "evidence from the reviews most relevant to the persona, and provide a "
    "1-10 suitability rating with a short justification grounded in repeated "
    "customer feedback? Reject any response that omits a populated 'Summary:' line "
    "or fails to state 'Suitability: <score>/10' followed by an evidence-based "
    "justification."
)
reference = "\n".join(first_item.get("reference_output", [])) or None

# Load content_A from results/v1_test_baseline.json
with open("results/v1_test_baseline.json", "r", encoding="utf-8") as f:
    baseline_data = json.load(f)
content_A = baseline_data[0]

# Load content_B from results/v1_test_sft.json
with open("results/v1_test_sft.json", "r", encoding="utf-8") as f:
    sft_data = json.load(f)
content_B = sft_data[0]

# Load content_C from results/v1_test_pe.json
with open("results/v1_test_pe.json", "r", encoding="utf-8") as f:
    pe_data = json.load(f)
content_C = pe_data[0]

# Load content_D from results/v1_test_rl.json
with open("results/v1_test_rl.json", "r", encoding="utf-8") as f:
    rl_data = json.load(f)
content_D = rl_data[0]

# Response format must be consistent with RL process (renderers.Message dict)
response_A = [{"role": "assistant", "content": content_A}]
response_B = [{"role": "assistant", "content": content_B}]
response_C = [{"role": "assistant", "content": content_C}]
response_D = [{"role": "assistant", "content": content_D}]

async def main():
    tokenizer = get_tokenizer(RM_MODEL_NAME_FOR_TOKENIZER)
    renderer = renderers.get_renderer(RM_RENDERER_NAME, tokenizer=tokenizer)
    service_client = ServiceClient(base_url=BASE_URL)
    sampling_kwargs = {}
    sampling_kwargs["base_model"] = RM_MODEL_NAME_FOR_TOKENIZER
    preference_sampling_client = service_client.create_sampling_client(**sampling_kwargs)
    reward_model = PrometheusEvalPreferenceModelFromChatRenderer(renderer, preference_sampling_client, temperature=RM_TEMPERATURE)

    # Prepare all comparisons
    comparison_AC = PrometheusEvalComparison(
        prompt_conversation=prompt_conversation,
        completion_A=response_C,  # PE
        completion_B=response_A,  # Baseline
        rubric=rubric,
        reference=reference,
    )
    comparison_AB = PrometheusEvalComparison(
        prompt_conversation=prompt_conversation,
        completion_A=response_B,  # SFT
        completion_B=response_A,  # Baseline
        rubric=rubric,
        reference=reference,
    )
    comparison_AD = PrometheusEvalComparison(
        prompt_conversation=prompt_conversation,
        completion_A=response_D,  # RL
        completion_B=response_A,  # Baseline
        rubric=rubric,
        reference=reference,
    )
    comparison_BC = PrometheusEvalComparison(
        prompt_conversation=prompt_conversation,
        completion_A=response_C,  # PE
        completion_B=response_B,  # SFT
        rubric=rubric,
        reference=reference,
    )
    comparison_BD = PrometheusEvalComparison(
        prompt_conversation=prompt_conversation,
        completion_A=response_D,  # RL
        completion_B=response_B,  # SFT
        rubric=rubric,
        reference=reference,
    )
    comparison_CD = PrometheusEvalComparison(
        prompt_conversation=prompt_conversation,
        completion_A=response_D,  # RL
        completion_B=response_C,  # PE
        rubric=rubric,
        reference=reference,
    )

    # Run all comparisons in parallel
    tasks = [
        reward_model(comparison_AC),
        reward_model(comparison_AB),
        reward_model(comparison_AD),
        reward_model(comparison_BC),
        reward_model(comparison_BD),
        reward_model(comparison_CD),
    ]
    results = await asyncio.gather(*tasks)
    score_AC, score_AB, score_AD, score_BC, score_BD, score_CD = results

    # Print results
    print(f"PE vs Baseline Score: {score_AC}")
    if score_AC > 0:
        print("Baseline better than PE")
    elif score_AC < 0:
        print("PE better than Baseline")
    else:
        print("Tie")

    print(f"SFT vs Baseline Score: {score_AB}")
    if score_AB > 0:
        print("Baseline better than SFT")
    elif score_AB < 0:
        print("SFT better than Baseline")
    else:
        print("Tie")

    print(f"RL vs Baseline Score: {score_AD}")
    if score_AD > 0:
        print("Baseline better than RL")
    elif score_AD < 0:
        print("RL better than Baseline")
    else:
        print("Tie")

    print(f"PE vs SFT Score: {score_BC}")
    if score_BC > 0:
        print("SFT better than PE")
    elif score_BC < 0:
        print("PE better than SFT")
    else:
        print("Tie")

    print(f"RL vs SFT Score: {score_BD}")
    if score_BD > 0:
        print("SFT better than RL")
    elif score_BD < 0:
        print("RL better than SFT")
    else:
        print("Tie")

    print(f"RL vs PE Score: {score_CD}")
    if score_CD > 0:
        print("PE better than RL")
    elif score_CD < 0:
        print("RL better than PE")
    else:
        print("Tie")

if __name__ == "__main__":
    asyncio.run(main())
