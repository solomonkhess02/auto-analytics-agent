"""
LLM Client Factory.
"""

from config.settings import settings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


def get_llm(model_override: str = None):
    """
    Create and return an LLM client based on the active provider in .env

    Args:
        model_override: Optional model name to use instead of the default.

    Returns:
        A LangChain chat model.
    """
    provider = settings.llm_provider
    model = model_override or settings.llm_model

    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=0.1,
            google_api_key=settings.google_api_key,
        )

    elif provider == "openai":
        return ChatOpenAI(
            model=model,
            temperature=0.1,
            api_key=settings.openai_api_key,
        )

    elif provider == "ollama":
        return ChatOpenAI(
            model=model,
            temperature=0.1,
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )

    else:
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. "
            f"Set LLM_PROVIDER in .env to 'gemini', 'openai', or 'ollama'."
        )
