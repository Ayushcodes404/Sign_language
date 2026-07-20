"""
llm_backends.py

Pluggable "generation" step for the sign-language RAG pipeline. Pick a
provider at runtime via the LLM_PROVIDER environment variable:

    export LLM_PROVIDER=gemini      # Google Gemini API
    export LLM_PROVIDER=nim         # NVIDIA NIM (OpenAI-compatible endpoint)

Each provider needs its own API key env var (see below). Both functions
take the same input (a list of ASL gloss labels, e.g. ["HELLO", "MY",
"NAME"]) and return a natural-language sentence grounded only in those
signs.
"""

import os

from dotenv import load_dotenv

# Loads variables from a .env file in the current directory into the
# environment, if one exists. If no .env file is present, this does
# nothing -- os.environ still works as before via `export`.
load_dotenv()

SYSTEM_PROMPT = (
    "You are helping interpret American Sign Language. You will be given "
    "a sequence of signs detected in order, in ASL gloss notation (all "
    "caps, no grammar). Turn it into a natural, grammatically correct "
    "English sentence. Only use the signs given -- do not invent new "
    "content or add signs that weren't provided."
)


def _build_user_prompt(gloss_sequence):
    return f"Signs: {' '.join(gloss_sequence)}\n\nNatural English sentence:"


# ---------------------------------------------------------------------
# Gemini backend
# ---------------------------------------------------------------------
def _generate_gemini(gloss_sequence):
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_name,
        contents=f"{SYSTEM_PROMPT}\n\n{_build_user_prompt(gloss_sequence)}",
    )
    return response.text.strip()


# ---------------------------------------------------------------------
# NVIDIA NIM backend (OpenAI-compatible API)
# ---------------------------------------------------------------------
def _generate_nim(gloss_sequence):
    from openai import OpenAI

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY environment variable is not set.")

    model_name = os.environ.get("NIM_MODEL", "meta/llama-3.1-70b-instruct")
    base_url = os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")

    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(gloss_sequence)},
        ],
        max_tokens=100,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------
_PROVIDERS = {
    "gemini": _generate_gemini,
    "nim": _generate_nim,
}


def get_provider_name():
    """Read which provider to use from the LLM_PROVIDER env var."""
    provider = os.environ.get("LLM_PROVIDER", "gemini").lower().strip()
    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. Choose from: {list(_PROVIDERS)}"
        )
    return provider


def is_configured(provider):
    """Check whether the required API key for a provider is set."""
    if provider == "gemini":
        return bool(os.environ.get("GEMINI_API_KEY"))
    if provider == "nim":
        return bool(os.environ.get("NVIDIA_API_KEY"))
    return False


def glosses_to_sentence(gloss_sequence, provider=None):
    """
    Turn a list of ASL gloss labels into a natural sentence using
    whichever provider is configured (or explicitly passed in).
    """
    if not gloss_sequence:
        return "(no signs detected yet)"

    provider = provider or get_provider_name()
    if provider not in _PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'.")

    return _PROVIDERS[provider](gloss_sequence)