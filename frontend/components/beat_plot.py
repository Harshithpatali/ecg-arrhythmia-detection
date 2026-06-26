import plotly.express as px
import pandas as pd
import streamlit as st


def plot_heartbeat(
    beat,
):

    df = pd.DataFrame({

        "Sample":
            range(
                len(beat)
            ),

        "Amplitude":
            beat,
    })

    fig = px.line(

        df,

        x="Sample",

        y="Amplitude",

        title="Selected Heartbeat",
    )

    st.plotly_chart(

        fig,

        use_container_width=True,
    )