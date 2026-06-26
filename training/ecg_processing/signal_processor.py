"""
Signal processing.
"""

from training.preprocessing.heartbeat_segmenter import (
    HeartbeatSegmenter,
)

from training.preprocessing.normalizer import (
    HeartbeatNormalizer,
)


class SignalProcessor:

    def __init__(self):

        self.segmenter = (
            HeartbeatSegmenter()
        )

    def process(
        self,
        signal,
        annotations,
    ):

        beats, labels = (
            self.segmenter.segment(
                signal,
                annotations,
            )
        )

        beats = (
            HeartbeatNormalizer.normalize(
                beats
            )
        )

        return beats, labels