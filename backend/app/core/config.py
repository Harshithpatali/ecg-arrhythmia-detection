"""
Application settings.
"""

class Settings:

    APP_NAME = "ECG Arrhythmia Detection API"
    APP_VERSION = "1.0.0"
    API_PREFIX = "/api/v1"

    HOST = "0.0.0.0"
    PORT = 8000

    DEBUG = False
    FRONTEND_URL = "*"


settings = Settings()