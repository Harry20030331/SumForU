import argparse
import asyncio

import chz
import wandb
from dotenv import load_dotenv

from dataset import SFT_DIR
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig


def build_config(
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
    dataset_path: str = str((SFT_DIR / "v1_synthesized_output.jsonl").resolve()),
    log_path: str = "/home/yumingfeng/repo/SumForU/results/sft_personalized_model",
    learning_rate: float = 2e-4,
    num_epochs: int = 10,
    eval_every: int = 8,
    max_length: int = 4096,
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
        file_path=dataset_path,
    )

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
        }
    ).make()
    

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an SFT model with configurable hyperparameters")
    parser.add_argument("--model-name", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument(
        "--dataset-path",
        default=str((SFT_DIR / "v1_synthesized_output.jsonl").resolve()),
        help="Path to the SFT conversation JSONL file",
    )
    parser.add_argument(
        "--log-path",
        default="/home/yumingfeng/repo/SumForU/results/sft_personalized_model",
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
    config = build_config(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        log_path=args.log_path,
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
