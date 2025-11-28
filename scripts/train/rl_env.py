"""Utilities for preference modeling and RL datasets used by RLAIF runs."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Sequence

import chz
import tinker
from tinker_cookbook import renderers
from tinker_cookbook.rl.preference_envs import PreferenceEnv, TournamentPattern, get_pairs
from tinker_cookbook.rl.types import (
    Env,
    EnvGroupBuilder,
    Metrics,
    RLDataset,
    RLDatasetBuilder,
    Trajectory,
)
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer
from tinker_cookbook.utils.misc_utils import safezip

from .prometheus_types import (
    LabeledPrometheusEvalComparison,
    PrometheusEvalComparison,
    PrometheusEvalComparisonRendererFromChatRenderer,
    PrometheusEvalPreferenceModel,
    PrometheusEvalPreferenceModelFromChatRenderer,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RL dataset helpers
# ---------------------------------------------------------------------------


@chz.chz
class HFPrometheusEvalDatasetBuilder():
    """Build datasets for training with PrometheusEvalPreferenceModel."""

    model_name_for_tokenizer: str
    renderer_name: str

    @property
    def tokenizer(self) -> Tokenizer:
        return get_tokenizer(self.model_name_for_tokenizer)

    @property
    def renderer(self) -> renderers.Renderer:
        return renderers.get_renderer(self.renderer_name, self.tokenizer)


class PrometheusEvalPairwisePreferenceDataset(RLDataset):
    def __init__(
        self,
        dataset_builder: HFPrometheusEvalDatasetBuilder,
        batch_size: int,
        preference_model: PrometheusEvalPreferenceModel,
        tournament_pattern: TournamentPattern = TournamentPattern.ALL_PAIRS_BOTH_WAYS,
        group_size: int = 4,
    ):
        self.dataset_builder = dataset_builder
        self.batch_size = batch_size
        self.preference_model = preference_model
        self.train_dataset, _ = self.dataset_builder.get_train_and_test_datasets()
        self.tournament_pattern = tournament_pattern
        self.group_size = group_size

    def get_batch(self, index: int) -> list[EnvGroupBuilder]:
        rows = self.train_dataset.select(
            range(index * self.batch_size, (index + 1) * self.batch_size)
        )
        builder_list = [
            PrometheusEvalPairwisePreferenceGroupBuilder(
                convo_prefix=row["prompt_conversation"],
                policy_renderer=self.dataset_builder.renderer,
                tournament_pattern=self.tournament_pattern,
                preference_model=self.preference_model,
                rubric=row["rubric"],
                reference=row["reference"],
                num_envs=self.group_size,
            )
            for row in rows
            if row is not None
        ]
        return builder_list

    def __len__(self) -> int:
        return len(self.train_dataset) // self.batch_size


@chz.chz
class PrometheusEvalPairwisePreferenceRLDatasetBuilder(RLDatasetBuilder):
    comparison_dataset_builder: HFPrometheusEvalDatasetBuilder
    rm_renderer_name: str
    rm_model_name_for_tokenizer: str
    batch_size: int
    tournament_pattern: TournamentPattern = TournamentPattern.ALL_PAIRS_BOTH_WAYS
    rm_model_path: str | None
    group_size: int
    base_url: str | None = None

    async def __call__(self) -> tuple[PrometheusEvalPairwisePreferenceDataset, None]:
        tokenizer = get_tokenizer(self.rm_model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.rm_renderer_name, tokenizer=tokenizer)
        service_client = tinker.ServiceClient(base_url=self.base_url)
        sampling_kwargs: dict[str, str] = {}
        if self.rm_model_path:
            sampling_kwargs["model_path"] = self.rm_model_path
        else:
            sampling_kwargs["base_model"] = self.rm_model_name_for_tokenizer

        preference_sampling_client = service_client.create_sampling_client(**sampling_kwargs)

        return PrometheusEvalPairwisePreferenceDataset(
            dataset_builder=self.comparison_dataset_builder,
            batch_size=self.batch_size,
            preference_model=PrometheusEvalPreferenceModelFromChatRenderer(
                convo_renderer=renderer,
                sampling_client=preference_sampling_client,
            ),
            tournament_pattern=self.tournament_pattern,
            group_size=self.group_size,
        ), None


def get_pairs(n: int, pattern: TournamentPattern) -> list[tuple[int, int]]:
    if pattern == TournamentPattern.ALL_PAIRS_BOTH_WAYS:
        return [(i, j) for i in range(n) for j in range(n) if i != j]
    if pattern == TournamentPattern.ALL_PAIRS_ONE_WAY:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    raise ValueError(f"Invalid tournament pattern: {pattern}")


@dataclass(frozen=True)
class PrometheusEvalPairwisePreferenceGroupBuilder(EnvGroupBuilder):
    convo_prefix: list[renderers.Message]
    policy_renderer: renderers.Renderer
    tournament_pattern: TournamentPattern
    preference_model: PrometheusEvalPreferenceModel
    rubric: str
    reference: str | None
    num_envs: int

    async def make_envs(self) -> Sequence[Env]:
        return [
            PreferenceEnv(self.convo_prefix, self.policy_renderer) for _ in range(self.num_envs)
        ]

    def comparison_reward_for_second_messages(
        self, message_i: list[renderers.Message], message_j: list[renderers.Message]
    ) -> PrometheusEvalComparison:
        return PrometheusEvalComparison(
            prompt_conversation=self.convo_prefix,
            completion_A=[m for m in message_i],
            completion_B=[m for m in message_j],
            rubric=self.rubric,
        )

    async def get_response_message(self, trajectory: Trajectory) -> tuple[list[renderers.Message], bool]:
        response, is_valid = self.policy_renderer.parse_response(
            trajectory.transitions[0].ac.tokens
        )
        return [response], is_valid

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory]
    ) -> list[tuple[float, Metrics]]:
        assert all(len(trajectory.transitions) == 1 for trajectory in trajectory_group)
        response_tuples = await asyncio.gather(
            *[self.get_response_message(trajectory) for trajectory in trajectory_group]
        )
        response_messages, is_valid_list = safezip(*response_tuples)

        comparison_indices_pairs = get_pairs(len(response_messages), self.tournament_pattern)
        j_comparisons = [
            self.comparison_reward_for_second_messages(
                message_i=response_messages[i], message_j=response_messages[j]
            )
            for i, j in comparison_indices_pairs
        ]

        j_rewards = await asyncio.gather(*[self.preference_model(c) for c in j_comparisons])

        win_minus_loss_list = [0.0 for _ in range(len(response_messages))]
        matchup_count = [0 for _ in range(len(response_messages))]
        for (i, j), j_reward in safezip(comparison_indices_pairs, j_rewards):
            win_minus_loss_list[j] -= j_reward
            win_minus_loss_list[i] += j_reward
            matchup_count[j] += 1
            matchup_count[i] += 1
        format_coef = 1.0

        return [
            (
                (win_minus_loss / matchup if matchup else 0.0)
                + format_coef * (float(is_valid) - 1.0),
                {
                    "win_minus_loss": win_minus_loss / matchup if matchup else 0.0,
                    "format": is_valid,
                },
            )
            for win_minus_loss, is_valid, matchup in safezip(
                win_minus_loss_list, is_valid_list, matchup_count
            )
        ]
