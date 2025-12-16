# config.py
"""
Configuration file for PixStory Image Captioning experiments
"""
import torch
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULT_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Dataset
DATASET_SLUG = "adityajn105/flickr8k"
IMAGE_SIZE = 224
MAX_CAPTION_LEN = 25
MIN_WORD_FREQ = 5

# Training
BATCH_SIZE = 32
NUM_WORKERS = 4 if os.cpu_count() > 4 else 2
EPOCHS = 8
FINE_TUNE_EPOCHS = 6
LR = 1e-3
LR_FINE_TUNE = 1e-4
GRAD_CLIP = 5.0

# Model
EMBED_SIZE = 512
HIDDEN_SIZE = 512
HIDDEN_SIZE_SMALL = 256
DROPOUT = 0.3

# Special tokens
PAD_TOKEN = "<pad>"
START_TOKEN = "<start>"
END_TOKEN = "<end>"
UNK_TOKEN = "<unk>"

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Reproducibility
SEED = 42

# Experiment names (for saving)
EXP_RNN_TYPE = "rnn_type_comparison"
EXP_HIDDEN_SIZE = "hidden_size_comparison"
EXP_CNN_TUNING = "cnn_tuning_comparison"

def print_config():
    """Print configuration"""
    print("=" * 60)
    print("PIXSTORY IMAGE CAPTIONING - CONFIGURATION")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Embed size: {EMBED_SIZE}")
    print(f"Hidden size: {HIDDEN_SIZE}")
    print(f"Vocabulary min frequency: {MIN_WORD_FREQ}")
    print(f"Max caption length: {MAX_CAPTION_LEN}")
    print("=" * 60)