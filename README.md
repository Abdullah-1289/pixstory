# PixStory Image Captioning System

COSC 442 - Artificial Intelligence - Fall 2025

## 🎯 Project Overview
Image captioning system using CNN (ResNet-18) encoder and RNN (GRU/LSTM) decoder.

## 📊 Results
| Model | BLEU-1 | BLEU-4 | Status |
|-------|--------|--------|--------|
| GRU-512 (Frozen CNN) | 0.2904 | 0.0503 | ✅ Trained |
| LSTM-512 (Frozen CNN) | 0.2280 | 0.0375 | ✅ Trained |
| GRU-256 (Frozen CNN) | 0.2227 | 0.0354 | ✅ Trained |
| GRU-512 (Fine-tuned CNN) | 0.2349 | 0.0415 | ✅ Trained |

## 🏗️ Project Structure

pixstory/
├── experiments/ # Individual experiment scripts
│ ├── 01_rnn_type_comparison.py
│ ├── 02_hidden_size_comparison.py
│ └── 03_cnn_tuning_comparison.py
├── src/utils/ # Shared utilities
│ ├── data_loader.py
│ ├── models.py
│ ├── trainer.py
│ └── evaluator.py
├── models/ # Trained models
├── results/ # Results and logs
├── data/ # Dataset (auto-downloaded)
├── requirements.txt # Dependencies
├── setup.sh # Setup script
└── run_all_experiments.sh # Run all experiments
text


## 🚀 Quick Start
```bash
# 1. Setup
./setup.sh

# 2. Run experiments
./run_all_experiments.sh

# 3. Or run individual experiments
python experiments/01_rnn_type_comparison.py

📋 Requirements

    Python 3.8+

    PyTorch with CUDA support

    8GB+ GPU memory recommended

📄 License

Educational Use - COSC 442 Assignment
