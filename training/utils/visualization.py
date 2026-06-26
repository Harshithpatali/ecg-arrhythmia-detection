"""
Visualization utilities.
"""

import plotly.graph_objects as go
import numpy as np


def plot_ecg(
    signal: np.ndarray,
    title: str = "ECG Signal",
):
    """
    Create an interactive ECG plot.

    Parameters
    ----------
    signal : np.ndarray
        ECG signal.

    title : str
        Plot title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            y=signal,
            mode="lines",
            name="ECG",
        )
    )

    figure.update_layout(
        title=title,
        xaxis_title="Samples",
        yaxis_title="Amplitude",
        template="plotly_white",
        height=400,
    )

    return figure


def plot_heartbeat(
    beat: np.ndarray,
):
    """
    Plot a segmented heartbeat.

    Parameters
    ----------
    beat : np.ndarray

    Returns
    -------
    plotly.graph_objects.Figure
    """
    return plot_ecg(
        beat,
        title="Heartbeat",
    )