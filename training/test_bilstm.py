import tensorflow as tf

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from training.models.bilstm import (
    bilstm_feature_extractor,
)

inputs = Input(shape=(45, 128))

outputs = bilstm_feature_extractor(inputs)

model = Model(inputs, outputs)

model.summary()

x = tf.random.normal((4, 45, 128))

y = model(x)

print()

print("Input Shape :", x.shape)
print("Output Shape:", y.shape)