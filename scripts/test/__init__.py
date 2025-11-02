# __init__.py
"""
Initialize model samplers and handle prompt construction logic.
"""
import config

import json
import re
import os
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
    prompt_input = config.PROMPT_PATH if active_mode == config.JSON else None

    if active_mode == config.JSON:
        # check if file exists
        if not os.path.isfile(prompt_input):
            raise FileNotFoundError(f"JSON file not found: {prompt_input}")

        with open(prompt_input, "r", encoding="utf-8") as f:
            data = json.load(f)

        messages_list = []
        for i, item in enumerate(data):
            input_text = item.get("input", "")
            if config.USE_SYSTEM_PROMPT:
                messages = [
                    renderers.Message(role="system", content=config.SYSTEM_PROMPT),
                    renderers.Message(role="user", content=input_text),
                ]
            else:
                messages = [renderers.Message(role="user", content=input_text)]
            messages_list.append(messages)

        return messages_list

    elif active_mode == config.DIRECT:
        input_text = prompt_input
        if config.USE_SYSTEM_PROMPT:
            messages = [
                renderers.Message(role="system", content=config.SYSTEM_PROMPT),
                renderers.Message(role="user", content=input_text),
            ]
        else:
            messages = [renderers.Message(role="user", content=input_text)]

        return messages

    else:
        raise ValueError("Invalid active_mode. Choose from ['json', 'direct'].")
    

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

    print("✅ Config updated from command-line arguments:")
    print(f"  TEMPERATURE = {config.TEMPERATURE}")
    print(f"  MAX_TOKENS = {config.MAX_TOKENS}")
    print(f"  USE_SYSTEM_PROMPT = {config.USE_SYSTEM_PROMPT}")
