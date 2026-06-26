import streamlit as st


def render_about():

    st.title("About")

    st.write(
        """
        ECG Arrhythmia Detection
        using CNN, BiLSTM and
        Attention Mechanism.

        Dataset:
        MIT-BIH Arrhythmia Database.
        """
    )