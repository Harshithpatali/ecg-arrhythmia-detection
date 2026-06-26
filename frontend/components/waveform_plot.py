import pandas as pd
import plotly.express as px
import streamlit as st


def plot_waveform(
    signal,
    time_axis,
):

    df = pd.DataFrame({

        "Time":
            time_axis,

        "Amplitude":
            signal,
    })

    fig = px.line(

        df,

        x="Time",

        y="Amplitude",

        title="ECG Waveform",
    )

    st.plotly_chart(

        fig,

        use_container_width=True,
    )