import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 as cv

audio_ds = tf.io.read_file('test.wav')


audio_tensor, sample = tf.audio.decode_wav(audio_ds, desired_channels=1)
# print(audio_tensor)

tensor = tf.cast(audio_tensor, tf.float32)
tensor = tf.slice(tensor, [0, 0], [sample*2, -1])
plt.figure()
plt.plot(tensor.numpy())
plt.show()
print(tensor)
