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
EPOCHS = 20
LEARNING_RATE = 1e-3

# Model
LATENT_DIM = 16
KL_WEIGHT = 1.0