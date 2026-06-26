"""
PDF report generator.
"""

from io import BytesIO
from datetime import datetime

from reportlab.platypus import (
    SimpleDocTemplate,
    Spacer,
    Paragraph,
    PageBreak,
)

from reportlab.lib.styles import (
    getSampleStyleSheet,
)


class PDFReportGenerator:

    @staticmethod
    def generate(
        metadata,
        summary,
        confidences,
    ):

        buffer = BytesIO()

        doc = SimpleDocTemplate(
            buffer
        )

        styles = (
            getSampleStyleSheet()
        )

        elements = []

        elements.append(
            Paragraph(
                "ECG Arrhythmia Detection Report",
                styles["Title"],
            )
        )

        elements.append(
            Spacer(1, 20)
        )

        elements.append(
            Paragraph(
                f"Generated: {datetime.now()}",
                styles["Normal"],
            )
        )

        elements.append(
            Spacer(1, 20)
        )

        elements.append(
            Paragraph(
                "Record Information",
                styles["Heading2"],
            )
        )

        for key, value in metadata.items():

            elements.append(

                Paragraph(

                    f"{key}: {value}",

                    styles["Normal"],
                )
            )

        elements.append(
            Spacer(1, 20)
        )

        elements.append(
            Paragraph(
                "Risk Assessment",
                styles["Heading2"],
            )
        )

        elements.append(

            Paragraph(

                f"Risk Level: {summary['risk']['risk_level']}",

                styles["Normal"],
            )
        )

        elements.append(

            Paragraph(

                summary["risk"]["message"],

                styles["Normal"],
            )
        )

        elements.append(
            Spacer(1, 20)
        )

        elements.append(
            Paragraph(
                "Class Distribution",
                styles["Heading2"],
            )
        )

        for cls, value in summary[
            "distribution"
        ].items():

            elements.append(

                Paragraph(

                    f"{cls}: {value}",

                    styles["Normal"],
                )
            )

        elements.append(
            Spacer(1, 20)
        )

        elements.append(
            Paragraph(
                "Class Percentages",
                styles["Heading2"],
            )
        )

        for cls, value in summary[
            "percentages"
        ].items():

            elements.append(

                Paragraph(

                    f"{cls}: {value:.2f}%",

                    styles["Normal"],
                )
            )

        elements.append(
            PageBreak()
        )

        elements.append(
            Paragraph(
                "Confidence Statistics",
                styles["Heading2"],
            )
        )

        avg_conf = (
            sum(confidences)
            / len(confidences)
        )

        elements.append(

            Paragraph(

                f"Average Confidence: {avg_conf:.4f}",

                styles["Normal"],
            )
        )

        elements.append(
            Spacer(1, 20)
        )

        elements.append(
            Paragraph(
                "Model Information",
                styles["Heading2"],
            )
        )

        elements.append(

            Paragraph(

                "CNN + BiLSTM + Attention",

                styles["Normal"],
            )
        )

        elements.append(

            Paragraph(

                "Dataset: MIT-BIH Arrhythmia Database",

                styles["Normal"],
            )
        )

        doc.build(elements)

        buffer.seek(0)

        return buffer