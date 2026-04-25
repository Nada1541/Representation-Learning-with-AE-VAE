"""tf.data input pipeline for Medical MNIST."""

import os
import random

import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


def _list_region_files(data_root, region):
    region_dir = os.path.join(data_root, region)
    files = []
    for ext in ("*.jpeg", "*.jpg", "*.png"):
        files.extend(tf.io.gfile.glob(os.path.join(region_dir, ext)))
    if not files:
        raise FileNotFoundError(
            f"No images found in {region_dir}. "
            f"Check DATA_ROOT and that the region folder exists."
        )
    return sorted(files)


def _decode_image(path, image_size, channels):
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=channels, expand_animations=False)
    img.set_shape([None, None, channels])
    img = tf.image.resize(img, (image_size, image_size))
    img = tf.cast(img, tf.float32) / 255.0
    return img


def _to_pair(img):
    return img, img


def make_region_datasets(
    data_root,
    region,
    image_size=64,
    channels=1,
    batch_size=128,
    val_split=0.2,
    seed=42,
):
    """Build train/val tf.data.Datasets for a single region."""
    files = _list_region_files(data_root, region)
    random.Random(seed).shuffle(files)

    n_val = int(len(files) * val_split)
    val_files = files[:n_val]
    train_files = files[n_val:]

    def _build(file_list, training):
        ds = tf.data.Dataset.from_tensor_slices(file_list)
        if training:
            ds = ds.shuffle(
                buffer_size=len(file_list),
                seed=seed,
                reshuffle_each_iteration=True,
            )
        ds = ds.map(
            lambda p: _decode_image(p, image_size, channels),
            num_parallel_calls=AUTOTUNE,
        )
        ds = ds.map(_to_pair, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    return _build(train_files, training=True), _build(val_files, training=False)


def add_noise(images, noise_factor=0.3):
    noise = tf.random.normal(tf.shape(images), mean=0.0, stddev=noise_factor)
    return tf.clip_by_value(images + noise, 0.0, 1.0)