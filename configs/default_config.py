import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "ToothFairy_Dataset"
RAW_DATA_DIR = DATASET_DIR / "Dataset"
SPLITS_FILE = DATASET_DIR / "splits.json"

# Processed Cache Paths
PROCESSED_DIR = BASE_DIR / "processed_data"
DENSE_CACHE_DIR = PROCESSED_DIR / "dense_cached"
SPARSE_CACHE_DIR = PROCESSED_DIR / "sparse_cached"
PSEUDO_CACHE_DIR = PROCESSED_DIR / "pseudo_cached"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
LOGS_DIR = BASE_DIR / "logs"
OUTPUTS_DIR = BASE_DIR / "outputs"

for p in [PROCESSED_DIR, DENSE_CACHE_DIR, SPARSE_CACHE_DIR, PSEUDO_CACHE_DIR, CHECKPOINTS_DIR, LOGS_DIR, OUTPUTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Preprocessing Configs
TARGET_SPACING = (1.0, 1.0, 1.0)  # (z, y, x) isotropic in mm
INTENSITY_CLIPPING = (-500.0, 2500.0)  # HU window capturing canal and cortical bone
NUM_CLASSES = 3  # 0: Background, 1: Right IAN, 2: Left IAN (or 2 for binary foreground)
CLASS_NAMES = {0: "Background", 1: "Right_IAN", 2: "Left_IAN"}

# Training Hyperparameters (Tuned for 8GB RTX 4060)
PATCH_SIZE = (64, 128, 128)       # (D, H, W) - strictly memory safe on 8GB VRAM
BATCH_SIZE = 1                    # Per GPU step
GRADIENT_ACCUMULATION_STEPS = 2   # Effective batch size = 2
NUM_WORKERS = 2
LEARNING_RATE = 1e-2
WEIGHT_DECAY = 1e-4
MAX_EPOCHS_TEACHER = 300
MAX_EPOCHS_STUDENT = 350
NUM_FOLDS = 5

# Loss Weights
LOSS_WEIGHTS = {
    "ce": 0.40,
    "dice": 0.30,
    "cldice": 0.20,
    "boundary": 0.10,
}

# Geodesic & Pseudo-labeling Configs
PSEUDO_STABILITY_THRESHOLD = 0.88
PSEUDO_MIN_SPARSE_OVERLAP = 0.90
GEODESIC_RADIUS_MM = 2.5
