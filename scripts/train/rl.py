import asyncio
from dotenv import load_dotenv

import os, sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
parent_parent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
sys.path.append(parent_dir)
sys.path.append(parent_parent_dir)


import chz
from datasets import Dataset, concatenate_datasets, load_dataset
import numpy as np
import tinker
from tinker import types
from dataset import RL_DIR
from tinker_cookbook import cli_utils, model_info, renderers
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.rl import train

from scripts.train.rl_env import (
    PrometheusEvalPreferenceModelFromChatRenderer,
    PrometheusEvalComparison,
    HFPrometheusEvalDatasetBuilder,
    PrometheusEvalPairwisePreferenceRLDatasetBuilder,
)


class WinRateVsBaseOnTestEvaluator(SamplingClientEvaluator):
    """Measure policy win rate against the base model on the held-out split."""

    def __init__(
        self,
        comparison_dataset_builder: HFPrometheusEvalDatasetBuilder,
        policy_renderer_name: str,
        policy_tokenizer_model_name: str,
        reward_renderer_name: str,
        reward_tokenizer_model_name: str,
        reward_model_path: str | None,
        base_model_name: str,
        max_tokens: int,
        base_url: str | None = None,
    ):
        _train_ds, test_ds = comparison_dataset_builder.get_train_and_test_datasets()
        if test_ds is None:
            raise ValueError("Test dataset is not available for win-rate evaluation")
        self.test_ds = test_ds

        self.policy_renderer = renderers.get_renderer(
            policy_renderer_name, tokenizer=get_tokenizer(policy_tokenizer_model_name)
        )
        reward_renderer = renderers.get_renderer(
            reward_renderer_name, tokenizer=get_tokenizer(reward_tokenizer_model_name)
        )

        service_client = tinker.ServiceClient(base_url=base_url)
        reward_sampling_kwargs: dict[str, str] = {}
        if reward_model_path:
            reward_sampling_kwargs["model_path"] = reward_model_path
        else:
            reward_sampling_kwargs["base_model"] = reward_tokenizer_model_name

        reward_sampling_client = service_client.create_sampling_client(**reward_sampling_kwargs)
        self.preference_model = PrometheusEvalPreferenceModelFromChatRenderer(
            convo_renderer=reward_renderer,
            sampling_client=reward_sampling_client,
        )

        self.base_sampling_client = service_client.create_sampling_client(
            base_model=base_model_name
        )
        self.max_tokens = max_tokens

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        prompts = [example["prompt_conversation"] for example in self.test_ds]

        policy_sampling_params = types.SamplingParams(
            max_tokens=self.max_tokens,
            temperature=0.7,
            top_p=1.0,
            top_k=-1,
            stop=self.policy_renderer.get_stop_sequences(),
        )

        async def render_and_sample(
            client: tinker.SamplingClient, conversation: list[dict]
        ) -> str:
            convo_messages = [
                renderers.Message(role=msg["role"], content=msg["content"])
                for msg in conversation
            ]
            prompt = self.policy_renderer.build_generation_prompt(convo_messages)
            result = await client.sample_async(
                prompt=prompt,
                sampling_params=policy_sampling_params,
                num_samples=1,
            )
            sequence = result.sequences[0]
            return self.policy_renderer.tokenizer.decode(sequence.tokens).strip()

        async def get_responses(
            client: tinker.SamplingClient, prompt_conversations: list[list[dict]]
        ) -> list[str]:
            tasks = [render_and_sample(client, convo) for convo in prompt_conversations]
            return await asyncio.gather(*tasks)

        current_responses, base_responses = await asyncio.gather(
            get_responses(sampling_client, prompts),
            get_responses(self.base_sampling_client, prompts),
        )

        comparisons = [
            PrometheusEvalComparison(
                prompt_conversation=prompt,
                completion_A=[{"role": "assistant", "content": base}],
                completion_B=[{"role": "assistant", "content": cur}],
                rubric="Choose the better assistant response.",
            )
            for prompt, cur, base in zip(prompts, current_responses, base_responses)
        ]

        rm_results = await asyncio.gather(*[self.preference_model(c) for c in comparisons])

        num_wins = 0
        num_total = len(rm_results)
        for r in rm_results:
            if isinstance(r, (int, float)) and r > 0:
                num_wins += 1

        win_rate = num_wins / num_total if num_total > 0 else 0.0
        stderr = np.sqrt(win_rate * (1 - win_rate) / num_total) if num_total > 0 else 0.0

        return {"win_rate_vs_base": win_rate, "stderr_vs_base": stderr}


@chz.chz
class PrometheusEvalDatasetBuilderFromJSONL(HFPrometheusEvalDatasetBuilder):
    train_data_path: str
    test_data_path: str | None = None
    repeat_factor: int = 1

    def get_train_and_test_datasets(self) -> tuple[Dataset, Dataset | None]:
        train_ds = load_dataset("json", data_files=self.train_data_path, split="train")
        test_ds = (
            load_dataset("json", data_files=self.test_data_path, split="train")
            if self.test_data_path
            else None
        )

        def _prep(example: dict) -> dict:
            prompt_conversation = example["prompt_conversation"]
            reference = example.get("reference", None)
            rubric = example["rubric"]
            return {
                "prompt_conversation": prompt_conversation,
                "reference": reference,
                "rubric": rubric,
            }

        train_dataset = train_ds.map(_prep)
        test_dataset = test_ds.map(_prep) if test_ds is not None else None

        repeat_factor = max(1, self.repeat_factor)
        if repeat_factor > 1:
            train_dataset = concatenate_datasets([train_dataset] * repeat_factor)

        return train_dataset, test_dataset


def build_config(
    model_name: str,
    model_path: str | None,
    reward_model_name: str,
    reward_model_path: str | None,
    train_data_path: str,
    test_data_path: str,
    log_path: str,
    max_length: int,
    learning_rate: float,
    batch_size: int,
    group_size: int,
    eval_every: int,
    train_repeat: int,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
) -> train.Config:
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    comparison_dataset_builder = PrometheusEvalDatasetBuilderFromJSONL(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        train_data_path=train_data_path,
        test_data_path=test_data_path,
        repeat_factor=train_repeat,
    )
    builder = PrometheusEvalPairwisePreferenceRLDatasetBuilder(
        comparison_dataset_builder=comparison_dataset_builder,
        batch_size=batch_size,
        rm_renderer_name=model_info.get_recommended_renderer_name(reward_model_name),
        rm_model_name_for_tokenizer=reward_model_name,
        rm_model_path=reward_model_path,
        group_size=group_size,
    )

    def winrate_eval_builder():
        return WinRateVsBaseOnTestEvaluator(
            comparison_dataset_builder=comparison_dataset_builder,
            policy_renderer_name=renderer_name,
            policy_tokenizer_model_name=model_name,
            reward_renderer_name=model_info.get_recommended_renderer_name(reward_model_name),
            reward_tokenizer_model_name=reward_model_name,
            reward_model_path=reward_model_path,
            base_model_name=model_name,
            max_tokens=max_length,
        )

    return train.Config(
        model_name=model_name,
        log_path=log_path,
        dataset_builder=builder,
        learning_rate=learning_rate,
        max_tokens=max_length,
        eval_every=eval_every,
        evaluator_builders=[winrate_eval_builder],
        wandb_project=wandb_project,
        wandb_name=wandb_name,
        load_checkpoint_path=model_path,
    )


def main(
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
    model_path: str | None = None,
    reward_model_name: str = "Qwen/Qwen3-235B-A22B-Instruct-2507",
    reward_model_path: str | None = None,
    train_data_path: str = str((RL_DIR / "v1_rl_train.jsonl").resolve()),
    test_data_path: str = str((RL_DIR / "v1_rl_test.jsonl").resolve()),
    log_path: str = "results/logs/rl_4b_v3",
    max_length: int = 8192,
    learning_rate: float = 4e-5,
    batch_size: int = 16,
    group_size: int = 4,
    eval_every: int = 5,
    train_repeat: int = 5,   # repeat training data this many times, treat as an epoch multiplier
    wandb_project: str = "SumForU",
    wandb_name: str = "rl_4b_v3",
):
    load_dotenv()
    config = build_config(
        model_name=model_name,
        model_path=model_path,
        reward_model_name=reward_model_name,
        reward_model_path=reward_model_path,
        train_data_path=train_data_path,
        test_data_path=test_data_path,
        log_path=log_path,
        max_length=max_length,
        learning_rate=learning_rate,
        batch_size=batch_size,
        group_size=group_size,
        eval_every=eval_every,
        train_repeat=train_repeat,   
        wandb_project=wandb_project,
        wandb_name=wandb_name,
    )
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    chz.entrypoint(main, allow_hyphens=True)
