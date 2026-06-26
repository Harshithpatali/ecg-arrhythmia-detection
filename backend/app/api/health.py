from fastapi import APIRouter

from app.schemas.health import (
    HealthResponse
)

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
)
def health_check():

    return HealthResponse(

        status="healthy",

        api_version="1.0.0",

        service=(
            "ECG Arrhythmia API"
        ),
    )