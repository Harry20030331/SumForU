import json
import random
from pathlib import Path

from dataset import INTERIM_DIR
from scripts.test import config
from dataset.synthesize_data import load_persona_records, build_user_prompt

DEFAULT_INPUT_DIR = INTERIM_DIR.parent / "preprocessed"
DEFAULT_OUTPUT_DIR = INTERIM_DIR.parent / "processed"

def main():
    input_dir = DEFAULT_INPUT_DIR
    output_dir = DEFAULT_OUTPUT_DIR
    seed = 42

    input_files = list(input_dir.glob("preprocessed_*.json"))
    if not input_files:
        print(f"No preprocessed_*.json files found in {input_dir}")
        return

    print(f"Found {len(input_files)} categories to process.")

    for input_file in input_files:
        category = input_file.stem.replace("preprocessed_", "")
        records = load_persona_records(input_file)
        print(f"Loaded {len(records)} records for category {category}")

        random.seed(seed)
        shuffled = list(records)
        random.shuffle(shuffled)

        test_records = shuffled[300:400] if len(shuffled) >= 400 else []

        if test_records:
            test_sft_path = output_dir / "sft" / "test" / f"{category}.jsonl"
            test_sft_path.parent.mkdir(parents=True, exist_ok=True)
            with test_sft_path.open("w", encoding="utf-8") as fh:
                for record in test_records:
                    fh.write(json.dumps(record.__dict__, ensure_ascii=False) + "\n")
            print(f"Wrote {len(test_records)} test examples to {test_sft_path}")

if __name__ == "__main__":
    main()