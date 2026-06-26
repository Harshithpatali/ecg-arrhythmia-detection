"""
Application settings.
"""

from pydantic_settings import (
    BaseSettings
)


class Settings(
    BaseSettings
):

    APP_NAME: str = (
        "ECG Arrhythmia Detection API"
    )

    APP_VERSION: str = "1.0.0"

    API_PREFIX: str = "/api/v1"

    HOST: str = "0.0.0.0"

    PORT: int = 8000

    DEBUG: bool = False

    FRONTEND_URL: str = (
        "*"
    )

    class Config:

        env_file = ".env"


settings = Settings()