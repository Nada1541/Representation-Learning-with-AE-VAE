"""Project configuration: paths, dataset info, and hyperparameters."""

# Paths
DATA_ROOT = "/content/medical_mnist"
OUTPUT_DIR = "/content/drive/MyDrive/medical_mnist_outputs"

# Dataset
REGIONS = ["AbdomenCT", "BreastMRI", "ChestCT", "CXR", "Hand", "HeadCT"]
IMAGE_SIZE = 64
CHANNELS = 1
VAL_SPLIT = 0.2
SEED = 42

# Training
BATCH_SIZE = 128
EPOCHS = 30           # was 20 — more epochs help both models
LEARNING_RATE = 1e-3

# Model
LATENT_DIM = 32       # was 16 — more capacity = sharper recons
KL_WEIGHT = 0.001     # was 1.0 — much smaller β = sharper VAE images