import streamlit as st


def render_dashboard():

    st.title(
        "ECG Arrhythmia Detection"
    )

    st.markdown(
        """
        Deep Learning-Based
        Arrhythmia Classification

        CNN + BiLSTM + Attention
        """
    )