from configs.config import MITDB_DIR

from training.preprocessing.wfdb_loader import WFDBLoader
from training.preprocessing.annotation_parser import AnnotationParser
from training.preprocessing.heartbeat_segmenter import HeartbeatSegmenter
from training.preprocessing.normalizer import HeartbeatNormalizer


def main():
    loader = WFDBLoader(MITDB_DIR)

    record = loader.load_record("100")

    annotations = loader.load_annotations("100")

    parser = AnnotationParser()

    df = parser.parse(
        annotations["samples"],
        annotations["symbols"],
    )

    segmenter = HeartbeatSegmenter()

    beats, labels = segmenter.segment(
        record["signal"],
        df,
    )

    beats = HeartbeatNormalizer.normalize(beats)

    print(f"Signal shape: {record['signal'].shape}")
    print(f"Metadata: {record['metadata']}")
    print(f"Annotations: {len(df)}")
    print(f"Beats: {beats.shape}")
    print(f"Labels: {labels.shape}")
    print(df.head())


if __name__ == "__main__":
    main()
