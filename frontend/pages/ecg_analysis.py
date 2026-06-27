import streamlit as st
import pandas as pd

from components.upload import upload_ecg_files
from components.metrics import display_metrics
from components.charts import (
    plot_distribution,
    plot_percentages,
)

from utils.report_generator import (
    PDFReportGenerator
)

from services.api_client import ECGApiClient

from utils.helpers import average_confidence
from utils.constants import CLASS_NAMES


def render_ecg_analysis():

    st.title(
        "ECG Analysis"
    )

    st.markdown(
        """
        Upload MIT-BIH ECG files
        and analyze cardiac rhythm.
        """
    )

    uploaded_files = (
        upload_ecg_files()
    )

    if uploaded_files:

        st.success(
            f"{len(uploaded_files)} files uploaded."
        )

        if st.button(
            "Analyze ECG"
        ):

            with st.spinner(
                "Running inference..."
            ):

                try:

                    result = (
                        ECGApiClient.predict(
                            uploaded_files
                        )
                    )

                except Exception as e:

                    st.error(
                        str(e)
                    )

                    if (
                        hasattr(
                            e,
                            "response",
                        )
                        and e.response is not None
                    ):

                        st.code(
                            e.response.text
                        )

                    raise

            metadata = (
                result["metadata"]
            )

            summary = (
                result["summary"]
            )

            risk = (
                summary["risk"]
            )

            confidences = (
                result["confidences"]
            )

            avg_conf = (
                average_confidence(
                    confidences
                )
            )

            st.header(
                "Analysis Summary"
            )

            display_metrics(
                metadata,
                risk,
                avg_conf,
            )

            st.info(
                risk["message"]
            )

            st.divider()

            col1, col2 = (
                st.columns(2)
            )

            with col1:

                plot_distribution(
                    summary[
                        "distribution"
                    ]
                )

            with col2:

                plot_percentages(
                    summary[
                        "percentages"
                    ]
                )

            st.divider()

            st.header(
                "Class Percentages"
            )

            percentage_df = pd.DataFrame(
                summary[
                    "percentages"
                ].items(),
                columns=[
                    "Class",
                    "Percentage",
                ],
            )

            st.dataframe(
                percentage_df,
                use_container_width=True,
            )

            st.divider()

            st.header(
                "Sample Predictions"
            )

            sample_size = min(
                100,
                len(
                    result[
                        "beat_predictions"
                    ]
                ),
            )

            prediction_df = pd.DataFrame({

                "Beat":
                    range(
                        sample_size
                    ),

                "Prediction":
                    [
                        CLASS_NAMES.get(
                            x,
                            x,
                        )
                        for x in result[
                            "beat_predictions"
                        ][
                            :sample_size
                        ]
                    ],

                "Confidence":
                    result[
                        "confidences"
                    ][
                        :sample_size
                    ],
            })

            st.dataframe(
                prediction_df,
                use_container_width=True,
            )

            st.divider()

            st.header(
                "Download Report"
            )

            try:

                pdf_buffer = (
                    PDFReportGenerator.generate(
                        metadata,
                        summary,
                        confidences,
                    )
                )

                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_buffer,
                    file_name="ecg_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

            except Exception as e:

                st.error(
                    f"PDF generation failed: {e}"
                )

                st.exception(e)