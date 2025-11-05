import json
import os
import sys
import asyncio
from tqdm.asyncio import tqdm_asyncio 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tinker_cookbook import renderers
from scripts.test.utils import TinkerSampler, clear_content


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

    # Combine reviews
    combined_reviews = "\n".join(current_reviews + other_reviews)

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
    reviews = "\n\n".join(
        [f"Review {i+1}: {text.strip()}" for i, text in enumerate(review_texts)]
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


if __name__ == "__main__":
    input_path = "dataset/v1_test_stringified.json"
    output_path = "dataset/v1_test_preprocessed.json"
    asyncio.run(preprocess_reviews(input_path, output_path))
