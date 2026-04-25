"""Variational Autoencoder with the reparameterization trick."""

import tensorflow as tf
from tensorflow.keras import Model, layers, metrics


class Sampling(layers.Layer):
    """Reparameterization: z = mu + exp(0.5 * log_var) * epsilon."""

    def call(self, inputs):
        mu, log_var = inputs
        eps = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(0.5 * log_var) * eps


def build_vae_encoder(input_shape, latent_dim):
    inputs = layers.Input(shape=input_shape, name="encoder_input")
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    mu = layers.Dense(latent_dim, name="mu")(x)
    log_var = layers.Dense(latent_dim, name="log_var")(x)
    z = Sampling(name="z")([mu, log_var])
    return Model(inputs, [mu, log_var, z], name="vae_encoder")


def build_vae_decoder(latent_dim, output_channels=1):
    inputs = layers.Input(shape=(latent_dim,), name="decoder_input")
    x = layers.Dense(8 * 8 * 128, activation="relu")(inputs)
    x = layers.Reshape((8, 8, 128))(x)
    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
    outputs = layers.Conv2D(output_channels, 3, padding="same", activation="sigmoid")(x)
    return Model(inputs, outputs, name="vae_decoder")


class VAE(Model):
    def __init__(self, input_shape=(64, 64, 1), latent_dim=16, kl_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.encoder = build_vae_encoder(input_shape, latent_dim)
        self.decoder = build_vae_decoder(latent_dim, output_channels=input_shape[-1])

        self.total_loss_tracker = metrics.Mean(name="loss")
        self.recon_loss_tracker = metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.recon_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, x, training=False):
        _, _, z = self.encoder(x, training=training)
        return self.decoder(z, training=training)

    def encode(self, x):
        mu, _, _ = self.encoder(x, training=False)
        return mu

    def sample(self, n=16):
        z = tf.random.normal(shape=(n, self.latent_dim))
        return self.decoder(z, training=False)

    def _compute_losses(self, x, x_hat, mu, log_var):
        recon = tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.binary_crossentropy(x, x_hat), axis=(1, 2)
            )
        )
        kl = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=1)
        )
        total = recon + self.kl_weight * kl
        return total, recon, kl

    def train_step(self, data):
        x, _ = data
        with tf.GradientTape() as tape:
            mu, log_var, z = self.encoder(x, training=True)
            x_hat = self.decoder(z, training=True)
            total, recon, kl = self._compute_losses(x, x_hat, mu, log_var)
        grads = tape.gradient(total, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total)
        self.recon_loss_tracker.update_state(recon)
        self.kl_loss_tracker.update_state(kl)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, _ = data
        mu, log_var, z = self.encoder(x, training=False)
        x_hat = self.decoder(z, training=False)
        total, recon, kl = self._compute_losses(x, x_hat, mu, log_var)
        self.total_loss_tracker.update_state(total)
        self.recon_loss_tracker.update_state(recon)
        self.kl_loss_tracker.update_state(kl)
        return {m.name: m.result() for m in self.metrics}