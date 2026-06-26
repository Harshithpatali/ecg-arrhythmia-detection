"""
Frontend configuration.
"""

API_BASE_URL = (
    "http://localhost:8000"
)

PREDICT_ENDPOINT = (
    f"{API_BASE_URL}/api/v1/predict"
)

WAVEFORM_ENDPOINT = (
    f"{API_BASE_URL}/api/v1/waveform"
)

RPEAK_ENDPOINT = (
    f"{API_BASE_URL}/api/v1/rpeaks"
)

HEARTBEAT_ENDPOINT = (
    f"{API_BASE_URL}/api/v1/heartbeats"
)