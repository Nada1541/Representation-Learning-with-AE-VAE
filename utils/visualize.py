"""Plotting helpers: reconstructions, latent space, samples, loss curves."""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.decomposition import PCA


def plot_reconstructions(model, dataset, n=8, title=""):
    for batch in dataset.take(1):
        x, _ = batch
        x = x[:n]
        x_hat = model(x, training=False).numpy()
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
    for i in range(n):
        axes[0, i].imshow(x[i].numpy().squeeze(), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(x_hat[i].squeeze(), cmap="gray")
        axes[1, i].axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_latent_space(model, dataset, title="", max_points=2000):
    zs = []
    n_collected = 0
    for batch in dataset:
        x, _ = batch
        zs.append(model.encode(x).numpy())
        n_collected += zs[-1].shape[0]
        if n_collected >= max_points:
            break
    zs = np.concatenate(zs, axis=0)[:max_points]
    if zs.shape[1] > 2:
        zs_2d = PCA(n_components=2).fit_transform(zs)
    else:
        zs_2d = zs
    plt.figure(figsize=(6, 6))
    plt.scatter(zs_2d[:, 0], zs_2d[:, 1], s=4, alpha=0.5)
    plt.title(title)
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_generated_samples(vae_model, n=16, title="VAE samples"):
    samples = vae_model.sample(n).numpy()
    grid = int(np.ceil(np.sqrt(n)))
    fig, axes = plt.subplots(grid, grid, figsize=(grid * 1.5, grid * 1.5))
    for i in range(grid * grid):
        ax = axes.flat[i]
        if i < n:
            ax.imshow(samples[i].squeeze(), cmap="gray")
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_loss_history(history, title=""):
    plt.figure(figsize=(8, 4))
    for k, v in history.history.items():
        plt.plot(v, label=k)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_denoising(model, dataset, noise_factor=0.3, n=8, title=""):
    for batch in dataset.take(1):
        x, _ = batch
        x = x[:n]
    noisy = tf.clip_by_value(
        x + tf.random.normal(tf.shape(x), stddev=noise_factor), 0.0, 1.0
    )
    denoised = model(noisy, training=False).numpy()
    fig, axes = plt.subplots(3, n, figsize=(2 * n, 6))
    for i in range(n):
        axes[0, i].imshow(x[i].numpy().squeeze(), cmap="gray"); axes[0, i].axis("off")
        axes[1, i].imshow(noisy[i].numpy().squeeze(), cmap="gray"); axes[1, i].axis("off")
        axes[2, i].imshow(denoised[i].squeeze(), cmap="gray"); axes[2, i].axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()