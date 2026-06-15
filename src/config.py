"""Shared configuration for the go-to-sleep persona-audit experiment.

All model IDs were verified live against the OpenRouter /models endpoint at
runtime. Keep these pinned for reproducibility (provider versions may rotate).
"""
import os

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS_PATH = os.path.join(WORKSPACE, "datasets", "sleep_persona_probe", "prompts.jsonl")
RESULTS_DIR = os.path.join(WORKSPACE, "results")
MODEL_OUT_DIR = os.path.join(RESULTS_DIR, "model_outputs")
FIG_DIR = os.path.join(WORKSPACE, "figures")
LOG_DIR = os.path.join(WORKSPACE, "logs")

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_KEY = os.environ.get("OPENROUTER_KEY") or os.environ.get("OPENROUTER_API_KEY")

# Cross-provider panel for the "how unified across models" question.
# label -> OpenRouter model id
MODELS = {
    "gpt-5.4-mini":      "openai/gpt-5.4-mini",
    "claude-sonnet-4.6": "anthropic/claude-sonnet-4.6",
    "gemini-3.5-flash":  "google/gemini-3.5-flash",
    "llama-3.3-70b":     "meta-llama/llama-3.3-70b-instruct",
    "deepseek-v3.2":     "deepseek/deepseek-v3.2",
}

# Independent judge model (different role than the panel; temperature 0).
JUDGE_MODEL = "openai/gpt-5.4-mini"

# Generation parameters
N_REPLICATES = 3
GEN_TEMPERATURE = 0.7
GEN_MAX_TOKENS = 400
SEED_BASE = 42  # replicate r uses seed SEED_BASE + r

# Concurrency
MAX_CONCURRENCY = 12
REQUEST_TIMEOUT = 90.0
MAX_RETRIES = 5

for d in (RESULTS_DIR, MODEL_OUT_DIR, FIG_DIR, LOG_DIR):
    os.makedirs(d, exist_ok=True)
