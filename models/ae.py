"""Convolutional Autoencoder for 64x64 grayscale medical images."""

import tensorflow as tf
from tensorflow.keras import Model, layers


def build_encoder(input_shape, latent_dim):
    inputs = layers.Input(shape=input_shape, name="encoder_input")
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    z = layers.Dense(latent_dim, name="latent")(x)
    return Model(inputs, z, name="ae_encoder")


def build_decoder(latent_dim, output_channels=1):
    inputs = layers.Input(shape=(latent_dim,), name="decoder_input")
    x = layers.Dense(8 * 8 * 128, activation="relu")(inputs)
    x = layers.Reshape((8, 8, 128))(x)
    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
    outputs = layers.Conv2D(
        output_channels, 3, padding="same", activation="sigmoid", name="reconstruction"
    )(x)
    return Model(inputs, outputs, name="ae_decoder")


class Autoencoder(Model):
    def __init__(self, input_shape=(64, 64, 1), latent_dim=16, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.encoder = build_encoder(input_shape, latent_dim)
        self.decoder = build_decoder(latent_dim, output_channels=input_shape[-1])

    def call(self, x, training=False):
        z = self.encoder(x, training=training)
        return self.decoder(z, training=training)

    def encode(self, x):
        return self.encoder(x, training=False)