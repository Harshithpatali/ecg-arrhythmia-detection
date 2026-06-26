"""
Singleton model loader.
"""

from tensorflow.keras.models import load_model

from configs.config import MODEL_PATH

from training.models.attention import (
    AttentionLayer,
)


class ModelLoader:

    _model = None

    @classmethod
    def get_model(cls):

        if cls._model is None:

            cls._model = load_model(

                MODEL_PATH,

                custom_objects={

                    "AttentionLayer":
                    AttentionLayer

                }

            )

        return cls._model