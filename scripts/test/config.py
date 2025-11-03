# config.py
"""
Configuration file storing model paths, sampling parameters, and prompt settings.
"""
DIRECT = "direct"
JSON = "json"
BASELINE = "baseline"
SFT = "sft"
DPO = "dpo"

# Common generation parameters
TEMPERATURE = 1.0
MAX_TOKENS = 2048

# Model names (from Tinker or Hugging Face)
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

# Tinker model paths
SFT_MODEL_PATH = "tinker://059111f6-6682-45e6-89e6-30f04b35bb20/sampler_weights/final"
DPO_MODEL_PATH = "tinker://1ac4b6e1-095d-493f-83d5-5ab4067e651f/sampler_weights/final"

# Prompt settings
PROMPT_MODE = DIRECT  # Options: 'direct' or 'json'
USER_PROMPT = "Explain the difference between supervised fine-tuning and DPO in simple terms."
PROMPT_PATH = "data.json"  # Used if PROMPT_MODE is 'json'

# System prompt settings
USE_SYSTEM_PROMPT = False
SYSTEM_PROMPT = "Print Thank you before you explain the difference."

