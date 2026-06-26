"""
API communication layer for the ECG Arrhythmia Detection frontend.
"""

import requests

from services.config import (
    PREDICT_ENDPOINT,
    WAVEFORM_ENDPOINT,
    RPEAK_ENDPOINT,
    HEARTBEAT_ENDPOINT,
)


class ECGApiClient:
    """
    Client for making API calls to the backend.
    All methods accept a list of uploaded files (Streamlit UploadedFile objects).
    """

    @staticmethod
    def _prepare_files(uploaded_files):
        """
        Convert a list of Streamlit uploaded files into a list of tuples
        suitable for a multipart/form-data POST request.

        Args:
            uploaded_files: List of UploadedFile objects

        Returns:
            List of (field_name, (filename, file_content)) tuples
        """
        files = []
        for file in uploaded_files:
            files.append(
                (
                    "files",  # field name expected by the backend
                    (file.name, file.getvalue()),
                )
            )
        return files

    @classmethod
    def _post_request(cls, endpoint, uploaded_files, timeout=300):
        """
        Private helper to send a POST request with files to a given endpoint.

        Args:
            endpoint: str - the URL to POST to
            uploaded_files: list - files to upload
            timeout: int - request timeout in seconds

        Returns:
            dict - parsed JSON response

        Raises:
            requests.HTTPError: if the request fails
        """
        files = cls._prepare_files(uploaded_files)
        response = requests.post(endpoint, files=files, timeout=timeout)
        response.raise_for_status()
        return response.json()

    @classmethod
    def predict(cls, uploaded_files):
        """Send files for arrhythmia classification."""
        return cls._post_request(PREDICT_ENDPOINT, uploaded_files)

    @classmethod
    def waveform(cls, uploaded_files):
        """Get the raw ECG waveform data."""
        return cls._post_request(WAVEFORM_ENDPOINT, uploaded_files)

    @classmethod
    def rpeaks(cls, uploaded_files):
        """Get the R-peak positions detected in the signals."""
        return cls._post_request(RPEAK_ENDPOINT, uploaded_files)

    @classmethod
    def heartbeats(cls, uploaded_files):
        """Get the segmented heartbeats from the signals."""
        return cls._post_request(HEARTBEAT_ENDPOINT, uploaded_files)