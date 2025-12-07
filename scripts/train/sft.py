import json
import argparse
import asyncio

import chz
import wandb
import tinker
import re
from dotenv import load_dotenv

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from scripts.test import config
from scripts.test.utils import build_user_prompt

from dataset import SFT_DIR
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

import asyncio
from pathlib import Path
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from scripts.eval._eval_summaries_multi import (
    calc_references, calc_rouge, calc_bleu, calc_bertscore,
    calc_distinct, calc_usr, calc_entropy, calc_review_tokens, calc_persona_tokens, calc_coverage,
    calc_suitability_scores, calc_avg_ratings, calc_align_and_filter, calc_score_metrics, calc_classification_metrics
)

def load_test_prompts_and_refs(json_path: str) -> tuple[list, list]:
    """
    Load prompts and reference responses from a test set JSON file.
    Each item should contain a 'prompt' field and a 'reference_output' field (list).
    Returns (prompts, ref_response).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    messages_list = []
    for i, item in enumerate(data):
        input_text = build_user_prompt(item["persona"], item["reviews"])
        messages_list.append(config.SYSTEM_PROMPT + "\n\n" + input_text)

    ref_response = [item.get("reference_output", []) for item in data]
    return messages_list, ref_response

class SummarizationMetricsEvaluator(SamplingClientEvaluator):
    """
    Evaluates summarization model outputs on a labeled test set using multiple metrics.
    """

    def __init__(
        self,
        test_data_path: str,
        model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
        metrics: list[str] = None,
        sample_limit: int = None,
    ):
        self.metrics = metrics or ["rouge", "bleu", "bertscore", "distinct", "coverage", "score_metrics", "classification_metrics"]
        self.sample_limit = sample_limit
        # Load prompts and reference responses from the test set JSON file
        # import pdb; pdb.set_trace()
        self.prompts, self.ref_response = load_test_prompts_and_refs(test_data_path)
        self.prompts = self.prompts[:sample_limit] if sample_limit else self.prompts
        self.ref_response = self.ref_response[:sample_limit] if sample_limit else self.ref_response
        # Load and cache ground truth data for metrics
        with open(test_data_path, 'r', encoding='utf-8') as f:
            self.gt_data = json.load(f)
        self.gt_data = self.gt_data[:sample_limit] if sample_limit else self.gt_data

        self.tokenizer = get_tokenizer(model_name)

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict:
        """
        Generate model responses for each prompt using get_response, then calculate metrics.
        """
        # Helper: generate a response for a single prompt
        async def get_response(prompt_text: str) -> str:
            # Query the model for a personalized product summary
            response = await sampling_client.sample_async(
                prompt=tinker.types.ModelInput.from_ints(
                    self.tokenizer.encode(prompt_text)
                ),
                sampling_params=tinker.types.SamplingParams(
                    max_tokens=256, temperature=0.0
                ),
                num_samples=1,
            )
            decoded = self.tokenizer.decode(response.sequences[0].tokens).strip()
            return decoded

        # Use asyncio.gather for concurrent generation
        preds = await asyncio.gather(*[get_response(prompt) for prompt in self.prompts])

        results = {}

        # import pdb; pdb.set_trace()
        # 3. Calculate metrics based on generated predictions
        if "rouge" in self.metrics:
            results.update(calc_rouge(preds, self.ref_response))
        if "bleu" in self.metrics:
            results["bleu4"] = calc_bleu(preds, self.ref_response)
        if "bertscore" in self.metrics:
            p, r, f1 = calc_bertscore(preds, self.ref_response)
            results.update({"bertscore_p": p, "bertscore_r": r, "bertscore_f1": f1})
        if "distinct" in self.metrics:
            results["distinct_2"] = calc_distinct(preds, 2)
            results["distinct_3"] = calc_distinct(preds, 3)
            results["usr"] = calc_usr(preds)
            results["entropy"] = calc_entropy(preds)
        if "coverage" in self.metrics:
            review_vocab = calc_review_tokens(self.gt_data)
            persona_vocab = calc_persona_tokens(self.gt_data)
            results["review_coverage"] = calc_coverage(preds, review_vocab)
            results["persona_coverage"] = calc_coverage(preds, persona_vocab)
        if "score_metrics" in self.metrics or "classification_metrics" in self.metrics:
            pred_scores = calc_suitability_scores(preds)
            gt_scores = calc_avg_ratings(self.gt_data)
            if pred_scores is not None:
                pred_arr, gt_arr = calc_align_and_filter(pred_scores, gt_scores)
                if "score_metrics" in self.metrics:
                    results.update(calc_score_metrics(pred_arr, gt_arr))
                if "classification_metrics" in self.metrics:
                    results.update(calc_classification_metrics(pred_arr, gt_arr))

        return results
    
def build_config(
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
    trainset_path: str = str((SFT_DIR / "v1_synthesized_output.jsonl").resolve()),
    testset_path: str = "dataset/data/raw/v1_test_preprocessed.json",
    log_path: str = "results/logs/sft_personalized_model",
    learning_rate: float = 2e-4,
    num_epochs: int = 10,
    eval_every: int = 8,
    max_length: int = 8192,
    batch_size: int = 16,
    lr_schedule: str = "linear",
    wandb_project: str = "SumForU",
    wandb_name: str = "sft_4b_v1",
) -> train.Config:
    """
    Build supervised fine-tuning (SFT) config with wandb logging enabled.
    """
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=max_length,
        batch_size=batch_size,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    dataset = FromConversationFileBuilder(
        common_config=common_config, 
        file_path=trainset_path,
    )

    def summarization_eval_builder():
        return SummarizationMetricsEvaluator(
            test_data_path=testset_path,
            model_name=model_name,
            metrics=["rouge", "bleu", "bertscore", "distinct", "coverage", "score_metrics", "classification_metrics"],
            sample_limit=100,  # Optional
        )
    evaluator_builders = [summarization_eval_builder]
    
    return chz.Blueprint(train.Config).apply(
        {
            "log_path": log_path,
            "model_name": model_name,
            "dataset_builder": dataset,
            "learning_rate": learning_rate,
            "lr_schedule": lr_schedule,
            "num_epochs": num_epochs,
            "eval_every": eval_every,
            "wandb_project": wandb_project,
            "wandb_name": wandb_name,
            "evaluator_builders": evaluator_builders,
        }
    ).make()
    

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an SFT model with configurable hyperparameters")
    parser.add_argument("--model-name", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument(
        "--trainset-path",
        default=str((SFT_DIR / "v1_synthesized_output.jsonl").resolve()),
        help="Path to the SFT conversation JSONL file",
    )
    parser.add_argument(
        "--testset-path",
        default="dataset/data/raw/v1_test_preprocessed.json",
        help="Path to the SFT test set JSON file",
    )
    parser.add_argument(
        "--log-path",
        default="results/logs/sft_personalized_model",
        help="Directory where checkpoints and logs will be written",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=8)
    parser.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help="Maximum token length per sample; set lower for short prompts to save memory",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--lr-schedule",
        default="linear",
        help="Learning rate schedule string accepted by tinker_cookbook",
    )
    parser.add_argument("--wandb-project", default="SumForU")
    parser.add_argument("--wandb-name", default="sft_4b_v1")
    return parser.parse_args(argv)


def main():
    load_dotenv()
    wandb.login()
    args = parse_args()
    # Ensure all important paths are absolute
    trainset_path = str(Path(args.trainset_path).resolve())
    testset_path = str(Path(args.testset_path).resolve())
    log_path = str(Path(args.log_path).resolve())

    config = build_config(
        model_name=args.model_name,
        trainset_path=trainset_path,
        testset_path=testset_path,
        log_path=log_path,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        eval_every=args.eval_every,
        max_length=args.max_length,
        batch_size=args.batch_size,
        lr_schedule=args.lr_schedule,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
    )
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))

if __name__ == "__main__":
    main()
