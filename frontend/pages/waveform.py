import streamlit as st

from frontend.components.upload import (
    upload_ecg_files,
)

from frontend.components.waveform_plot import (
    plot_waveform,
)

from frontend.components.beat_plot import (
    plot_heartbeat,
)

from frontend.services.api_client import (
    ECGApiClient,
)


def render_waveform():

    st.title(
        "Waveform Viewer"
    )

    uploaded_files = (
        upload_ecg_files()
    )

    if uploaded_files:

        if st.button(
            "Load ECG"
        ):

            with st.spinner(
                "Loading waveform..."
            ):

                waveform = (

                    ECGApiClient.waveform(
                        uploaded_files
                    )

                )

                rpeaks = (

                    ECGApiClient.rpeaks(
                        uploaded_files
                    )

                )

                beats = (

                    ECGApiClient.heartbeats(
                        uploaded_files
                    )

                )

            st.success(
                "Waveform Loaded"
            )

            st.subheader(
                "ECG Signal"
            )

            plot_waveform(

                waveform["signal"],

                waveform["time_axis"],
            )

            st.metric(

                "Detected R-Peaks",

                rpeaks[
                    "total_peaks"
                ],
            )

            st.divider()

            beat_index = (

                st.slider(

                    "Select Heartbeat",

                    0,

                    beats[
                        "total_beats"
                    ] - 1,

                    0,
                )

            )

            selected_beat = (

                beats[
                    "beats"
                ][
                    beat_index
                ]
            )

            plot_heartbeat(
                selected_beat
            )