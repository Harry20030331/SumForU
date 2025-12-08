import argparse
import asyncio
import glob
import json
import os
import sys
from typing import Sequence

from tqdm.asyncio import tqdm_asyncio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset import INTERIM_DIR
from scripts.test.utils import TinkerSampler, clear_content
from tinker_cookbook import renderers


async def create_persona(current_reviews, other_reviews):
    """
    Generate a persona string based on product reviews.
    """
    model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    sampler = TinkerSampler(
        model_name=model_name,
        temperature=0.7,
        max_tokens=512,
        top_p=0.9,
        top_k=-1,
    )

    # Limit the number of reviews to avoid exceeding context window
    max_reviews_per_type = 10  # limit number of reviews per type
    max_review_length = 500  # characters
    
    def truncate_reviews(reviews, max_count, max_length):
        truncated = []
        for review in reviews[:max_count]:
            truncated.append(review[:max_length] + "..." if len(review) > max_length else review)
        return truncated
    
    current_reviews_limited = truncate_reviews(current_reviews, max_reviews_per_type, max_review_length)
    other_reviews_limited = truncate_reviews(other_reviews, max_reviews_per_type, max_review_length)
    
    # Combine reviews
    combined_reviews = "\n".join(current_reviews_limited + other_reviews_limited)

    SYSTEM_PROMPT = """You are an expert market analyst specializing in customer personas.
                       Generate the persona text directly, without any prefix or explanation.
                       Make the persona concise and focused on key traits, in ONLY ONE sentence."""

    USER_PROMPT = f"""
        Below are examples of how to generate customer personas based on product reviews.
        
        Example 1:
        Reviews:
        - "The product arrived quickly and was packaged securely."
        - "Very satisfied with the quality, feels durable and well-made."
        Persona:
        This customer values product quality and reliable delivery speed. They are less sensitive to price and prioritize a smooth, trustworthy shopping experience.
        
        Example 2:
        Reviews:
        - "It’s affordable and works well for daily use."
        - "Good deal for the price, I’ll buy again during discounts."
        Persona:
        This customer is price-conscious and tends to look for good value and deals. They enjoy functional products that balance cost and performance.
        
        ---
        
        Now, based on the following product reviews, create a concise persona of the typical customer:
        
        {combined_reviews}
        
        Provide insights into their purchase preferences.
        Please don't generate any persona descriptions not related to purchase preferences, like age, gender, or privacy.
        Generate the persona text directly, without any prefix, label, or explanation.
    """

    prompt_messages = [
        renderers.Message(role="system", content=SYSTEM_PROMPT),
        renderers.Message(role="user", content=USER_PROMPT)
    ]

    response = await sampler.generate(prompt_messages)
    persona = response['content'].strip()
    persona = clear_content(persona)

    # Debug print
    # print("--------------------------------------------------")
    # print("Reviews:\n", combined_reviews)
    # print("Generated Persona:\n", persona)
    # print("--------------------------------------------------\n")

    return persona

async def process_item(item):
    """Process one item asynchronously."""
    review_texts = item.get("input", [])
    # Limit the number of reviews
    max_reviews_num = 20  # limit number of reviews per type
    max_review_length = 500  # characters
    # Truncate review_texts to max_reviews_num
    review_texts = review_texts[:max_reviews_num]
    # Truncate each review text
    truncated_review_texts = [text.strip()[:max_review_length] + "..." if len(text.strip()) > max_review_length else text.strip() for text in review_texts]
    reviews = "\n\n".join(
        [f"Review {i+1}: {text}" for i, text in enumerate(truncated_review_texts)]
    )
    current_reviews = item["output"].get("current_product_reviews", [])
    other_reviews = item["output"].get("other_product_reviews", [])
    persona = await create_persona(current_reviews, other_reviews)
    return {
        "reviews": reviews,
        "persona": persona,
        "reference_output": current_reviews,
    }
    
async def preprocess_reviews(input_path, output_path):
    """
    Asynchronously process all reviews and generate personas concurrently.
    """
    if not os.path.exists(input_path):
        print(f"Input file does not exist: {input_path}")
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("================================================")
    print(f"Processing {len(data)} items from: {input_path}")
    print("================================================")
    
    processed = []

    tasks = [asyncio.create_task(process_item(item)) for item in data]
    processed = await tqdm_asyncio.gather(*tasks)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)

    print("================================================")
    print(f"Processed file successfully saved to: {output_path}")
    print("================================================")


async def batch_preprocess(input_dir, output_dir):
    """
    Batch process all stringified JSON files in input_dir and save to output_dir.
    Collect all tasks and process them in parallel with a unified progress bar.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all stringified_*.json files
    pattern = os.path.join(input_dir, "stringified_*.json")
    input_files = glob.glob(pattern)
    
    if not input_files:
        print(f"No stringified_*.json files found in {input_dir}")
        return
    
    print(f"Found {len(input_files)} files to process.")
    
    # Collect all tasks and map them
    all_tasks = []
    task_map = {}  # key: task_index, value: (file_index, item_index)
    file_data = []  # list of (input_file, data, output_file)
    
    for file_idx, input_file in enumerate(input_files):
        # Extract category from filename
        basename = os.path.basename(input_file)
        category = basename.replace("stringified_", "").replace(".json", "")
        output_file = os.path.join(output_dir, f"preprocessed_{category}.json")
        
        # Load data
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        file_data.append((input_file, data, output_file))
        
        # Create tasks for this file
        for item_idx, item in enumerate(data):
            task = asyncio.create_task(process_item(item))
            task_index = len(all_tasks)
            all_tasks.append(task)
            task_map[task_index] = (file_idx, item_idx)
    
    print(f"Total tasks to process: {len(all_tasks)}")
    
    # Process all tasks in parallel with unified progress bar
    processed_results = await tqdm_asyncio.gather(*all_tasks, desc="Processing all items")
    
    # Group results by file
    file_results = [[] for _ in file_data]
    for task_index, result in enumerate(processed_results):
        file_idx, item_idx = task_map[task_index]
        file_results[file_idx].append((item_idx, result))
    
    # Save results for each file
    for file_idx, (input_file, original_data, output_file) in enumerate(file_data):
        results = file_results[file_idx]
        # Sort by item_idx to maintain order
        results.sort(key=lambda x: x[0])
        processed = [res for _, res in results]
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(processed, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(processed)} items to {output_file}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess persona review dataset")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str((INTERIM_DIR.parent / "raw").resolve()),
        help="Directory containing stringified JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str((INTERIM_DIR.parent / "preprocessed").resolve()),
        help="Directory to save preprocessed JSON files",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(batch_preprocess(args.input_dir, args.output_dir))
