import streamlit as st


def render_sidebar():

    st.sidebar.title(
        "ECG Arrhythmia AI"
    )

    st.sidebar.markdown(
        """
        Upload ECG records and
        detect cardiac arrhythmias
        using CNN + BiLSTM + Attention.
        """
    )

    st.sidebar.info(
        """
        Supported Files:

        • .dat

        • .hea

        • .atr (optional)
        """
    )