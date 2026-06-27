"""
Frontend configuration.
"""

API_BASE_URL = (
    "https://ecg-arrhythmia-detection-k4z0.onrender.com"
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