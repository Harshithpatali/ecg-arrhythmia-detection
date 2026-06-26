import pandas as pd
import plotly.express as px
import streamlit as st


def plot_distribution(
    distribution,
):

    df = pd.DataFrame({

        "Class":
            list(
                distribution.keys()
            ),

        "Count":
            list(
                distribution.values()
            ),
    })

    fig = px.bar(

        df,

        x="Class",

        y="Count",

        title="Arrhythmia Distribution",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


def plot_percentages(
    percentages,
):

    df = pd.DataFrame({

        "Class":
            list(
                percentages.keys()
            ),

        "Percentage":
            list(
                percentages.values()
            ),
    })

    fig = px.pie(

        df,

        names="Class",

        values="Percentage",

        title="Class Distribution %",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )