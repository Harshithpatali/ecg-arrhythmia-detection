from configs.config import MITDB_DIR

from training.preprocessing.wfdb_loader import (
    WFDBLoader,
)

from training.preprocessing.annotation_parser import (
    AnnotationParser,
)

from training.preprocessing.heartbeat_segmenter import (
    HeartbeatSegmenter,
)

from training.preprocessing.normalizer import (
    HeartbeatNormalizer,
)

from training.inference.beat_predictor import (
    BeatPredictor,
)

loader = WFDBLoader(MITDB_DIR)

record = loader.load_record("100")

annotations = loader.load_annotations("100")

parser = AnnotationParser()

df = parser.parse(

    annotations["samples"],

    annotations["symbols"]

)

segmenter = HeartbeatSegmenter()

beats, labels = segmenter.segment(

    record["signal"],

    df

)

beats = HeartbeatNormalizer.normalize(
    beats
)

predictor = BeatPredictor()

result = predictor.predict(
    beats[0]
)

print(result)