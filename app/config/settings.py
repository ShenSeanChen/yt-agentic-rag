# Copyright 2024
# Directory: yt-agentic-rag/app/config/settings.py

"""
Application Settings - Environment Variable Management.

Loads and validates all configuration from environment variables:
- Supabase credentials for vector database
- OpenAI/Anthropic API keys for LLM
- Google service account for calendar/email tools
- Application settings (log level, environment, etc.)

Uses Pydantic Settings for automatic .env file loading and validation.
"""

from typing import Literal, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Required environment variables:
    - SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_SERVICE_ROLE_KEY
    - OPENAI_API_KEY
    
    Optional environment variables:
    - ANTHROPIC_API_KEY (for Claude support)
    - GOOGLE_SERVICE_ACCOUNT_PATH, GOOGLE_CALENDAR_EMAIL (for tools)
    """
    
    # =========================================================================
    # Supabase Configuration
    # =========================================================================
    supabase_url: str = Field(..., env="SUPABASE_URL")
    supabase_anon_key: str = Field(..., env="SUPABASE_ANON_KEY")
    supabase_service_role_key: str = Field(..., env="SUPABASE_SERVICE_ROLE_KEY")
    
    # =========================================================================
    # AI Provider Configuration
    # =========================================================================
    ai_provider: Literal["openai", "anthropic"] = Field(
        default="openai", 
        env="AI_PROVIDER"
    )
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_embed_model: str = Field(
        default="text-embedding-3-small", 
        env="OPENAI_EMBED_MODEL"
    )
    openai_chat_model: str = Field(
        default="gpt-4o", 
        env="OPENAI_CHAT_MODEL"
    )
    
    # Anthropic Configuration (optional)
    anthropic_api_key: str = Field(default="", env="ANTHROPIC_API_KEY")
    anthropic_chat_model: str = Field(
        default="claude-3-5-sonnet-20240620", 
        env="ANTHROPIC_CHAT_MODEL"
    )
    
    # =========================================================================
    # Google API Configuration (for Agentic Tools)
    # =========================================================================
    # Path to Google service account JSON file
    google_service_account_path: str = Field(
        default="credentials/service_account.json",
        env="GOOGLE_SERVICE_ACCOUNT_PATH"
    )
    
    # Email address for Google Calendar and Gmail (domain-wide delegation)
    # This is the email that will create calendar events and send emails
    google_calendar_email: str = Field(
        default="",
        env="GOOGLE_CALENDAR_EMAIL"
    )
    
    # Calendar ID to create events on (use 'primary' for main calendar)
    google_calendar_id: str = Field(
        default="primary",
        env="GOOGLE_CALENDAR_ID"
    )
    
    # =========================================================================
    # Application Configuration
    # =========================================================================
    environment: str = Field(default="development", env="ENVIRONMENT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # =========================================================================
    # RAG Configuration
    # =========================================================================
    default_top_k: int = Field(default=6)
    chunk_size: int = Field(default=400)  # Approximate tokens per chunk
    chunk_overlap: int = Field(default=60)  # 15% overlap between chunks
    temperature: float = Field(default=0.1)  # LLM temperature
    embedding_dimensions: int = Field(default=1536)  # text-embedding-3-small
    
    # =========================================================================
    # Agent Configuration
    # =========================================================================
    # Maximum iterations for the agent loop to prevent infinite loops
    max_agent_iterations: int = Field(default=5)
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """
    Get application settings instance.
    
    Returns:
        Settings instance with all configuration values
    """
    return settings
