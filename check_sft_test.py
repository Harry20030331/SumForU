import json
from pathlib import Path

def main():
    test_dir = Path("dataset/data/processed/sft/test")
    if not test_dir.exists():
        print(f"Directory {test_dir} does not exist.")
        return

    jsonl_files = list(test_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No .jsonl files found in {test_dir}")
        return

    for file_path in jsonl_files:
        print(f"\nChecking {file_path.name}:")
        count = 0
        valid = 0
        with file_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                count += 1
                try:
                    sample = json.loads(line.strip())
                    # Check if sample has persona, reviews, reference_output
                    persona = sample.get("persona", "")
                    reviews = sample.get("reviews", "")
                    reference_output = sample.get("reference_output", [])
                    if persona.strip() and reviews.strip() and reference_output:
                        valid += 1
                    else:
                        if not persona.strip():
                            print(f"  Sample {count}: persona is empty")
                        if not reviews.strip():
                            print(f"  Sample {count}: reviews is empty")
                        if not reference_output:
                            print(f"  Sample {count}: reference_output is empty")
                except json.JSONDecodeError:
                    print(f"  Sample {count}: invalid JSON")
        print(f"  Total samples: {count}")
        print(f"  Valid samples: {valid}")

if __name__ == "__main__":
    main()