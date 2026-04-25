import torch

# ======================
# Paths
# ======================
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
CHECKPOINT_DIR = "outputs/checkpoints"
LOG_DIR = "outputs/logs"
FIGURES_DIR = "outputs/figures"

# ======================
# Training parameters
# ======================
BATCH_SIZE = 16
NUM_EPOCHS = 1
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# Model parameters
# ======================
MODEL_NAME = "mobilenet_v2"
DATASET_NAME = "concrete_crack"
NUM_CLASSES = 2  # Crack / No Crack

# ======================
# Data augmentation
# ======================
IMAGE_SIZE = 224

# ======================
# Random seed for reproducibility
# ======================
SEED = 42

# ======================
# Task type
# ======================
TASK = "classification"  # or "segmentation"