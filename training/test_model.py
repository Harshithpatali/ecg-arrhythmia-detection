from training.models.cnn_bilstm_attention import (
    build_model,
)

model = build_model()

model.summary()