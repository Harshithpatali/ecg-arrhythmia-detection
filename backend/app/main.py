from fastapi import FastAPI

from fastapi.middleware.cors import (
    CORSMiddleware
)

from app.core.config import (
    settings
)

from app.core.logging import (
    configure_logging
)

from app.api.health import (
    router as health_router
)
from app.api.prediction import (
    router as prediction_router
)

from app.api.waveform import (
    router as waveform_router
)

from app.api.rpeaks import (
    router as rpeak_router
)

from app.api.heartbeats import (
    router as heartbeat_router
)


logger = configure_logging()

app = FastAPI(

    title=settings.APP_NAME,

    version=settings.APP_VERSION,
)

app.add_middleware(

    CORSMiddleware,

    allow_origins=[
        settings.FRONTEND_URL
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)

app.include_router(

    health_router,

    prefix=settings.API_PREFIX,

    tags=["Health"],
)

app.include_router(

    prediction_router,

    prefix=settings.API_PREFIX,

    tags=["Prediction"],
)

app.include_router(
    waveform_router,
    prefix=settings.API_PREFIX,
    tags=["Waveform"],
)

app.include_router(
    rpeak_router,
    prefix=settings.API_PREFIX,
    tags=["RPeaks"],
)

app.include_router(
    heartbeat_router,
    prefix=settings.API_PREFIX,
    tags=["Heartbeats"],
)



@app.get("/")
def root():

    return {

        "message":
            "ECG Arrhythmia API",

        "version":
            settings.APP_VERSION,
    }