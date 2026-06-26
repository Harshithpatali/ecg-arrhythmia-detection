"""
Risk analysis.
"""


class RiskAnalyzer:

    HIGH_RISK = [
        "V",
        "F",
    ]

    MEDIUM_RISK = [
        "S",
    ]

    @staticmethod
    def assess(
        class_percentages
    ):

        ventricular = (
            class_percentages.get(
                "V",
                0,
            )
        )

        fusion = (
            class_percentages.get(
                "F",
                0,
            )
        )

        supraventricular = (
            class_percentages.get(
                "S",
                0,
            )
        )

        if ventricular > 10:

            return {
                "risk_level":
                    "High",
                "message":
                    "Frequent ventricular arrhythmias detected."
            }

        if fusion > 5:

            return {
                "risk_level":
                    "High",
                "message":
                    "Fusion beats exceed threshold."
            }

        if supraventricular > 15:

            return {
                "risk_level":
                    "Medium",
                "message":
                    "Frequent supraventricular arrhythmias detected."
            }

        return {
            "risk_level":
                "Low",
            "message":
                "Predominantly normal rhythm."
        }