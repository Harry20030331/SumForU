import chz
import wandb
from dotenv import load_dotenv
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig


def build_config(
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
    dataset_path: str = "/home/yumingfeng/repo/SumForU/dataset/v1_synthesized_output.jsonl",
    log_path: str = "/home/yumingfeng/repo/SumForU/results/sft_personalized_model",
    learning_rate: float = 2e-4,
    num_epochs: int = 10,
    eval_every: int = 8,
    wandb_project: str = "SumForU_SFT",
    wandb_name: str = "4b_v1",
) -> train.Config:
    """
    Build supervised fine-tuning (SFT) config with wandb logging enabled.
    """
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=32768,
        batch_size=64,
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
            "lr_schedule": "linear",
            "num_epochs": num_epochs,
            "eval_every": eval_every,
            "wandb_project": wandb_project,
            "wandb_name": wandb_name,
        }
    ).make()
    

def main():
    load_dotenv()
    wandb.login()
    config = build_config()
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    train.main(config)

if __name__ == "__main__":
    main()
