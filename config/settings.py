"""
Application settings using Pydantic Settings.
All configuration is loaded from environment variables / .env file.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """Central configuration for the Auto-Analytics Agent."""

    # --- LLM ---
    llm_provider: str = Field(default="openai", description="LLM provider: openai, gemini, ollama")

    # OpenAI
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o", description="OpenAI model name")

    # Google Gemini
    google_api_key: Optional[str] = Field(default=None, description="Google AI API key")
    gemini_model: str = Field(default="gemini-1.5-pro", description="Gemini model name")

    # Ollama (local)
    ollama_model: str = Field(default="llama3", description="Ollama model name")

    # --- Application ---
    app_env: str = Field(default="development", description="Environment: development, production")
    debug: bool = Field(default=True, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")

    # --- Backend ---
    backend_host: str = Field(default="0.0.0.0", description="Backend host")
    backend_port: int = Field(default=8000, description="Backend port")

    # --- Frontend ---
    frontend_port: int = Field(default=8501, description="Streamlit port")
    backend_url: str = Field(default="http://localhost:8000", description="Backend URL for frontend")

    # --- Code Execution ---
    code_execution_timeout: int = Field(default=120, description="Timeout in seconds for code execution")
    max_retries: int = Field(default=3, description="Max retries for self-healing")

    # --- Data Limits ---
    max_upload_size_mb: int = Field(default=100, description="Max upload size in MB")
    max_rows_for_profiling: int = Field(default=50000, description="Max rows to send for profiling")

    @property
    def llm_model(self) -> str:
        """Return the model name for the active provider."""
        provider_model_map = {
            "openai": self.openai_model,
            "gemini": self.gemini_model,
            "ollama": self.ollama_model,
        }
        return provider_model_map.get(self.llm_provider, self.openai_model)

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


# Singleton instance — import this throughout the project
settings = Settings()
