"""
Global constants used throughout the project.
"""

# -----------------------------------
# ECG
# -----------------------------------

SAMPLING_RATE = 360

BEAT_LENGTH = 180

LEFT_WINDOW = 90

RIGHT_WINDOW = 90

# -----------------------------------
# Dataset
# -----------------------------------

RANDOM_STATE = 42

TEST_SIZE = 0.10

VALIDATION_SIZE = 0.10

# -----------------------------------
# Training
# -----------------------------------

BATCH_SIZE = 256

EPOCHS = 10

LEARNING_RATE = 1e-3

# -----------------------------------
# Classes
# -----------------------------------

AAMI_CLASSES = [
    "N",
    "S",
    "V",
    "F",
    "Q",
]

CLASS_DESCRIPTIONS = {
    "N": "Normal",
    "S": "Supraventricular",
    "V": "Ventricular",
    "F": "Fusion",
    "Q": "Unknown",
}

# -----------------------------------
# WFDB Labels
# -----------------------------------

ANNOTATION_MAP = {
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",

    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",

    "V": "V",
    "E": "V",

    "F": "F",

    "/": "Q",
    "f": "Q",
    "Q": "Q",
}