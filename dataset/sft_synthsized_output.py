import json
import os
import sys
import asyncio
from tqdm.asyncio import tqdm_asyncio
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tinker_cookbook import renderers
from scripts.test.utils import TinkerSampler, clear_content
from scripts.test import config


async def process_item(message: dict) -> dict:
    """
    Generate a summary in a 'messages' format for instruction tuning.
    """
    model_name = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    sampler = TinkerSampler(
        model_name=model_name,
        temperature=0.7,
        max_tokens=512,
        top_p=0.9,
        top_k=-1,
    )

    SYSTEM_PROMPT = config.SYSTEM_PROMPT
    USER_PROMPT = config.USER_PROMPT
    USER_PROMPT = USER_PROMPT.replace("<reviews>", message.get("reviews", ""))
    USER_PROMPT = USER_PROMPT.replace("<persona>", message.get("persona", ""))

    # Append reference review instructions
    USER_PROMPT += """\n\n
            I also provide the real review written by the customer who has this persona and purchased the products.
            It will give a reference for you to better understand the persona and extract the key characteristics of the reviews.
            However, remember you are writing a summary based on the reviews provided earlier, so you need
            to ONLY focus on the persona and the reviews, not to generate similar reviews with the reference review.
            Here is the reference review:\n""" + "\n".join(message.get("reference_output", []))

    prompt_messages = [
        renderers.Message(role="system", content=SYSTEM_PROMPT),
        renderers.Message(role="user", content=USER_PROMPT)
    ]

    response = await sampler.generate(prompt_messages)
    summary = response['content'].strip()
    summary = clear_content(summary)

    # Build instruction-tuning style record
    result = {
        "messages": [
            {
                "role": "user",
                "content": SYSTEM_PROMPT + USER_PROMPT.strip()
            },
            {
                "role": "assistant",
                "content": summary
            }
        ]
    }

    return result


async def synthesize_output(input_path, output_path):
    """
    Asynchronously process all reviews and generate summaries concurrently,
    saving them in JSON Lines format (one item per line).
    """
    if not os.path.exists(input_path):
        print(f"Input file does not exist: {input_path}")
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("==================================================")
    print(f"Synthesizing {len(data)} summaries from: {input_path}")
    print("==================================================")

    tasks = [asyncio.create_task(process_item(item)) for item in data]
    processed = await tqdm_asyncio.gather(*tasks)

    # Write each result as a single line JSON (JSONL format)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in processed:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("==================================================")
    print(f"Synthesized summaries successfully saved to: {output_path}")
    print("==================================================")


if __name__ == "__main__":
    input_path = "dataset/v1_preprocessed.json"
    output_path = "dataset/v1_synthesized_output.jsonl"  # JSON Lines format
    asyncio.run(synthesize_output(input_path, output_path))
