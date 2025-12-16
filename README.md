```markdown
# PixStory Image Captioning Project

A CNN+RNN image captioning system for the PixStory case study (COSC 442).

## 📁 Project Structure

```
pixstory-project/
├── run_all_experiments.sh         # Run all 3 experiments
├── original_implementation.py     # Complete working implementation
├── requirements.txt               # Python dependencies
├── config.py                      # Configuration
├── experiments/                   # 3 separate experiments
│   ├── exp01_rnn_type.py         # LSTM vs GRU comparison
│   ├── exp02_hidden_size.py      # 256 vs 512 hidden units
│   └── exp03_cnn_tuning.py       # Frozen vs fine-tuned CNN
├── src/utils/                     # Core modules
│   ├── data_loader.py            # Dataset handling
│   ├── models.py                 # CNN encoder + RNN decoder
│   ├── trainer.py                # Training loop
│   └── evaluator.py              # BLEU evaluation
├── models/                        # Pre-trained models (4 files)
└── results/                       # Generated results
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run All Experiments
```bash
chmod +x run_all_experiments.sh
./run_all_experiments.sh
```

### 3. Run Individual Experiments
```bash
# Experiment 1: RNN Type Comparison
python3 experiments/exp01_rnn_type.py

# Experiment 2: Hidden Size Comparison  
python3 experiments/exp02_hidden_size.py

# Experiment 3: CNN Tuning Comparison
python3 experiments/exp03_cnn_tuning.py
```

## 🔬 Experiments

### Experiment 1: RNN Type Comparison
**File**: `experiments/exp01_rnn_type.py`  
**Goal**: Compare LSTM vs GRU decoders with frozen CNN encoder  
**Output**: BLEU scores and comparison plot

### Experiment 2: Hidden Size Comparison  
**File**: `experiments/exp02_hidden_size.py`  
**Goal**: Compare 256 vs 512 hidden units in RNN decoder  
**Output**: Performance vs computation trade-off analysis

### Experiment 3: CNN Tuning Comparison
**File**: `experiments/exp03_cnn_tuning.py`  
**Goal**: Compare frozen vs fine-tuned CNN encoder  
**Output**: Quality improvement vs overfitting analysis

## 📊 Results Summary (Pre-trained Models)

The `models/` folder contains 4 pre-trained models with the following performance:

| Model | Description | BLEU-1 | BLEU-4 |
|-------|-------------|---------|---------|
| `exp1_gru.pth` | GRU decoder, 512 hidden, frozen CNN | 0.2904 | 0.0503 |
| `exp1_lstm.pth` | LSTM decoder, 512 hidden, frozen CNN | 0.2280 | 0.0375 |
| `exp2_gru_256.pth` | GRU decoder, 256 hidden, frozen CNN | 0.2227 | 0.0354 |
| `exp3_finetuned.pth` | GRU decoder, 512 hidden, fine-tuned CNN | 0.2349 | 0.0415 |

**Note**: Experiments load these models if they exist, otherwise train new ones.

## 📈 Expected Output

Each experiment generates:
- CSV file with BLEU-1 and BLEU-4 scores in `results/`
- PNG plot visualizing comparisons in `results/`
- Console output with qualitative examples

## 💻 View Existing Results

To see the evaluation results that have already been computed:
```bash
cat results/model_evaluation_results.csv
```

## ⏱️ Time Estimates

- First run: Downloads Flickr8k dataset (~1GB)
- With GPU: ~30 minutes per experiment
- With CPU: ~2 hours per experiment
- Total: 1.5-6 hours depending on hardware

## 📝 Requirements

- Python 3.8+
- PyTorch 2.0+
- Kaggle account (dataset downloads automatically)

---
*Last updated: 2025-12-16*
```