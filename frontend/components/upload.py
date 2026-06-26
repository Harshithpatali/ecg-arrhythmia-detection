import streamlit as st


def upload_ecg_files():

    uploaded_files = st.file_uploader(

        "Upload ECG Record",

        type=[
            "dat",
            "hea",
            "atr",
        ],

        accept_multiple_files=True,
    )

    return uploaded_files