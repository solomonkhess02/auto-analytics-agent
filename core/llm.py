"""
LLM Client Factory — Creates connections to AI models.

WHAT THIS FILE DOES:
    Provides a single function `get_llm()` that every agent uses to get
    its LLM (Large Language Model) connection. The function reads from
    .env to decide which provider (Gemini, OpenAI, or Ollama) to use.

WHY A FACTORY:
    If we hardcode `ChatGoogleGenerativeAI(...)` in every agent, switching
    providers means changing every file. With a factory, we change one
    line in .env and all agents automatically switch.

USAGE:
    from core.llm import get_llm

    llm = get_llm()                         # Uses default from .env
    llm = get_llm("gemini-1.5-flash")       # Override with a specific model

    response = llm.invoke("What is 2+2?")   # Send a prompt
    print(response.content)                  # "4"
"""

from config.settings import settings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


def get_llm(model_override: str = None):
    """
    Create and return an LLM client based on the active provider in .env

    Args:
        model_override: Optional model name to use instead of the default.
                       Example: "gemini-1.5-flash" for a faster/cheaper model.

    Returns:
        A LangChain chat model with a .invoke() method.

    How agents use this:
        llm = get_llm()
        response = llm.invoke("Analyze this dataset...")
        answer = response.content  # The LLM's text response
    """

    provider = settings.llm_provider
    model = model_override or settings.llm_model

    # --- Gemini (Google) ---
    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=0.1,    # Low = consistent, predictable responses
            google_api_key=settings.google_api_key,
        )

    # --- OpenAI ---
    elif provider == "openai":
        return ChatOpenAI(
            model=model,
            temperature=0.1,
            api_key=settings.openai_api_key,
        )

    # --- Ollama (local, free, no API key needed) ---
    elif provider == "ollama":
        # Ollama exposes an OpenAI-compatible API on localhost
        # So we reuse ChatOpenAI but point it to the local server
        return ChatOpenAI(
            model=model,
            temperature=0.1,
            base_url="http://localhost:11434/v1",
            api_key="ollama",   # Ollama doesn't need a real key
        )

    else:
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. "
            f"Set LLM_PROVIDER in .env to 'gemini', 'openai', or 'ollama'."
        )
