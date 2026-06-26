import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from training.models.cnn import cnn_feature_extractor

inputs = Input(shape=(180, 1))

outputs = cnn_feature_extractor(inputs)

model = Model(inputs, outputs)

model.summary()

x = tf.random.normal((4, 180, 1))

y = model(x)

print("\nInput Shape :", x.shape)
print("Output Shape:", y.shape)