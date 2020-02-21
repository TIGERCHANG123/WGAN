import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class generator(tf.keras.Model):
  def __init__(self, input_shape, img_shape):
    super(generator, self).__init__()
    self.model = tf.keras.Sequential()

    self.model.add(tf.keras.layers.Dense(256, input_dim=input_shape))
    self.model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    self.model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    self.model.add(tf.keras.layers.Dense(512))
    self.model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    self.model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    self.model.add(tf.keras.layers.Dense(1024))
    self.model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    self.model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    self.model.add(tf.keras.layers.Dense(np.prod(img_shape), activation='tanh'))
    self.model.add(tf.keras.layers.Reshape(img_shape))

  def call(self, x):
    x = self.model(x)
    return x

class discriminator(tf.keras.Model):
  def __init__(self, img_shape):
    super(discriminator, self).__init__()
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Flatten(input_shape=img_shape))
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.Dense(256))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

    self.model = model
  def call(self, x):
    return self.model(x)

def get_gan(noise_shape, img_shape):
  Generator = generator(noise_shape[0], img_shape)
  Discriminator = discriminator(img_shape)
  gen_name = 'gan'
  return Generator, Discriminator, gen_name

