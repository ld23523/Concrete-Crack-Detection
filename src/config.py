import torch

# ======================
# Paths
# ======================
RAW_DATA_DIR = "../data/raw"
PROCESSED_DATA_DIR = "../data/processed"
CHECKPOINT_DIR = "../outputs/checkpoints"
LOG_DIR = "../outputs/logs"
FIGURES_DIR = "../outputs/figures"

# ======================
# Training parameters
# ======================
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# Model parameters
# ======================
MODEL_NAME = "mobilenet_v2"
NUM_CLASSES = 2  # Crack / No Crack

# ======================
# Data augmentation
# ======================
IMAGE_SIZE = 227  # Default input size for ResNet/EfficientNet

# ======================
# Random seed for reproducibility
# ======================
SEED = 42