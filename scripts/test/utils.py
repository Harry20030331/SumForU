# __init__.py
"""
Initialize model samplers and handle prompt construction logic.
"""
import config

import json
import re
import os
import re
import tinker
from dotenv import load_dotenv
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.tokenizer_utils import get_tokenizer

# Initialize Tinker Service Client
load_dotenv()
service_client = tinker.ServiceClient()

# Define TinkerSampler class
class TinkerSampler():
    """A simple wrapper around Tinker ServiceClient to do sampling."""
    def __init__(
        self,
        model_name: str,
        model_path: str | None = None,  # tinker://..., obtained from Tinker training job
        temperature: float = 0.9,
        max_tokens=1024,
        top_p=1,
        top_k=-1,  # -1 means no limit
    ):
        tokenizer = get_tokenizer(model_name)
        renderer_name = get_recommended_renderer_name(model_name)
        # Read https://tinker-docs.thinkingmachines.ai/rendering to understand what renderer is
        self.renderer = renderers.get_renderer(name=renderer_name, tokenizer=tokenizer)
        self.sampling_params = types.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=self.renderer.get_stop_sequences(),
        )
        self.sampling_client = service_client.create_sampling_client(
            model_path=model_path,
            base_model=model_name,
        )
        
    async def generate(self, messages: list[renderers.Message]) -> renderers.Message:
        prompt = self.renderer.build_generation_prompt(messages) #tokens is inside
        sampling_result = await self.sampling_client.sample_async(
            prompt=prompt,
            sampling_params=self.sampling_params,
            num_samples=1
        )
        response_tokens = sampling_result.sequences[0].tokens 

        raw_text_output = self.renderer.tokenizer.decode(response_tokens)
        response_message = {'role': 'assistant', 'content': raw_text_output}
        response_message['content'] = response_message['content'].strip()

        return response_message
    

def build_model(model_type: str) -> TinkerSampler:
    """
    Build and return a model sampler based on the selected model type.
    Args:
        model_type (str): One of ['baseline', 'sft', 'dpo'].
    Returns:
        TinkerSampler: Initialized sampler for the selected model.
    """
    model_name: str = config.MODEL_NAME
    if model_type == config.BASELINE:
        model_path = None
    elif model_type == config.SFT:
        model_path = config.SFT_MODEL_PATH
    elif model_type == config.DPO:
        model_path = config.DPO_MODEL_PATH
    else:
        raise ValueError("Invalid model_type. Choose from ['baseline', 'sft', 'dpo'].")

    return TinkerSampler(
        model_name=model_name,
        model_path=model_path,
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
    )


def build_prompt():
    """
    Build message(s) for model input depending on the active mode.

    Args:
        active_mode (str): One of ['json', 'direct'].
            - 'json': load prompts from a JSON file (each entry has 'input').
            - 'direct': use a single prompt string.
        prompt_input (str): Either a JSON file path or a single text prompt.

    Returns:
        list or list[list]: 
            - If 'direct', returns one message list.
            - If 'json', returns a list of message lists (one per item).
    """
    active_mode = config.PROMPT_MODE
    prompt_input = config.PROMPT_PATH if active_mode == config.JSON else config.USER_PROMPT

    messages_list = []

    if config.USE_SYSTEM_PROMPT:
        print("----------- Using system prompt -----------")
        print( config.SYSTEM_PROMPT)
        print("-------------------------------------------\n")

    if active_mode == config.JSON:
        # check if file exists
        if not os.path.isfile(prompt_input):
            raise FileNotFoundError(f"JSON file not found: {prompt_input}")

        with open(prompt_input, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"---------------------------------------")
        print(f"--- Loaded {len(data)} prompts from {prompt_input} ---")
        print(f"---------------------------------------\n")
        
        for i, item in enumerate(data):
            input_text = item.get("input", "")
            # print(f"Loaded prompt {i + 1}: {input_text}")
            if config.USE_SYSTEM_PROMPT:
                messages = [
                    renderers.Message(role="system", content=config.SYSTEM_PROMPT),
                    renderers.Message(role="user", content=input_text),
                ]
            else:
                messages = [renderers.Message(role="user", content=input_text)]
            messages_list.append(messages)

    elif active_mode == config.DIRECT:
        input_text = prompt_input
        if config.USE_SYSTEM_PROMPT:
            messages = [
                renderers.Message(role="system", content=config.SYSTEM_PROMPT),
                renderers.Message(role="user", content=input_text),
            ]
        else:
            messages = [renderers.Message(role="user", content=input_text)]
        messages_list.append(messages)
        
    else:
        raise ValueError("Invalid active_mode. Choose from ['json', 'direct'].")

    return messages_list


# update config by arguments
def update_config_from_args(args):
    """
    Update the config module variables based on command-line arguments.
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    # Update temperature
    if hasattr(args, "temperature") and args.temperature is not None:
        config.TEMPERATURE = args.temperature

    # Update max tokens
    if hasattr(args, "max_tokens") and args.max_tokens is not None:
        config.MAX_TOKENS = args.max_tokens

    # Update whether to use system prompt
    if hasattr(args, "use_system_prompt") and args.use_system_prompt is not None:
        # Convert string inputs like "true"/"false" to bool if needed
        if isinstance(args.use_system_prompt, str):
            args.use_system_prompt = args.use_system_prompt.lower() in ("true", "1", "yes")
        config.USE_SYSTEM_PROMPT = args.use_system_prompt

    # Update prompt or system prompt text if provided
    if hasattr(args, "user_prompt") and args.user_prompt:
        config.USER_PROMPT = args.user_prompt
    if hasattr(args, "system_prompt") and args.system_prompt:
        config.SYSTEM_PROMPT = args.system_prompt

    print("---------------------------------------")
    print("---         Config Updated!         ---")
    print("---------------------------------------")
    print()

def clear_content(s: str) -> str:
    """
    Clear unwanted patterns (e.g. <|im_end|>) from the content string.
    Args:
        s (str): Input string to be cleaned.]
    Returns:
        str: Cleaned string.
    """
    # Define unwanted patterns
    unwanted_patterns = [
        r"<\|im_end\|>",
        r"<\|endoftext\|>",
        r"\s+"
    ]
    for pattern in unwanted_patterns:
        s = re.sub(pattern, " ", s)
    return s.strip()

def pretty_model_output(text: str):
    """
    Auto-format long model outputs into readable multi-line sections, 
    while keeping Markdown tables intact.

    Improvements:
    - Detects only standalone '---' as section separators (not table lines)
    - Keeps Markdown table rows together
    - Preserves Markdown headers, bullet points, emojis, and short lines
    """

    # Insert newlines before Markdown headers and bullets, but not inside tables
    # text = re.sub(r"(?<!\|)(---)(?!\|)", r"\n\1\n", text)   # only standalone ---
    text = clear_content(text)
    text = re.sub(r"(### )", r"\n\1", text)
    text = re.sub(r"(\👉|\- |\* )", r"\n\1", text)
    text = re.sub(r"(\|.*\|)", r"\n\1", text)  # add breaks before table rows
    text = re.sub(r"\n{2,}", "\n", text)

    lines = text.strip().split("\n")

    print("=" * 32, " Model Output ", "=" * 32)

    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue

        # Section header
        if line.startswith("###"):
            print("\n" + "-" * 80)
            print(f"{i:02d} | {line}")
            print("-" * 80)

        # Real horizontal separator (standalone ---)
        elif re.fullmatch(r"-{3,}", line):
            print("-" * 80)

        # Markdown table rows
        elif "|" in line and re.match(r"^\|.*\|$", line):
            print(f"{i:02d} | {line}")

        # Bullets or subpoints
        elif re.match(r"^[-•*]\s", line.strip()) or line.strip().startswith("👉"):
            print(f"{i:02d} |   {line}")

        else:
            print(f"{i:02d} | {line}")

    print("=" * 80 + "\n\n\n")

def print_conversation(messages: list[dict], response: dict):
    """
    Print the conversation messages in a readable format.
    Args:
        messages (list[renderers.Message]): List of message objects.
    """
    input = messages[-1]["content"]
    output = response["content"]

    print("---------- User Input ----------")
    print(input)
    print("--------------------------------\n")
    pretty_model_output(output)
