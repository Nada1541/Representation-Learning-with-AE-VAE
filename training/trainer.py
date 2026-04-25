import os

import tensorflow as tf


def _make_callbacks(save_path):
    callbacks = []
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=save_path,
                save_best_only=True,
                save_weights_only=True,
                monitor="val_loss",
                verbose=0,
            )
        )
    return callbacks


def train_ae(model, train_ds, val_ds, epochs=20, lr=1e-3, save_path=None):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
    )
    return model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=_make_callbacks(save_path),
        verbose=2,
    )


def train_vae(model, train_ds, val_ds, epochs=20, lr=1e-3, save_path=None):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr))
    for x_batch, _ in train_ds.take(1):
        _ = model(x_batch, training=False)
        break

    return model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=_make_callbacks(save_path),
        verbose=2,
    )