import tensorflow as tf

from training.models.attention import AttentionLayer

x = tf.random.normal((8, 180, 128))

layer = AttentionLayer()

y = layer(x)

print("Input Shape :", x.shape)
print("Output Shape:", y.shape)