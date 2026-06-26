import streamlit as st


def display_metrics(
    metadata,
    risk,
    avg_confidence,
):

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    with col1:

        st.metric(
            "Sampling Rate",
            metadata["sampling_rate"],
        )

    with col2:

        st.metric(
            "Total Beats",
            metadata["total_beats"],
        )

    with col3:

        st.metric(
            "Risk Level",
            risk["risk_level"],
        )

    with col4:

        st.metric(
            "Avg Confidence",
            f"{avg_confidence:.2%}",
        )