# PixStory Image Captioning System

## 📊 Results Summary
- **Best BLEU-1**: 0.573 (Target: 0.40) ✓ **Exceeded by 43%**
- **Best BLEU-4**: 0.165 (Target: 0.10) ✓ **Exceeded by 65%**
- **Best Model**: Frozen ResNet-18 + GRU-256

## 🚀 Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

2. Run All Experiments

```bash

chmod +x run_all_experiments.sh
./run_all_experiments.sh
```
3. Generate Report Visualizations

```bash

python3 visualize_images.py    # 10 image+caption examples
python3 visualize_result.py    # Performance charts & tables
```

```📁 Project Structure

pixstory/
├── run_all_experiments.sh     # Run all experiments
├── run_rnn_only.py           # RNN: GRU vs LSTM
├── run_hidden_size.py        # Hidden: 256 vs 512
├── run_cnn_tuning.py         # CNN: Frozen vs Fine-tuned
├── rerun_finetuning.py       # 8-epoch test
├── visualize_images.py       # Generate image examples
├── visualize_result.py       # Generate charts/tables
├── original_implementation.py # Model architecture
├── config.py                 # Configuration
├── models/                   # Trained models (4 best)
├── results/                  # All outputs
└── src/utils/               # Utility modules
```

## 📈 Key Findings

    GRU (0.547) ≈ LSTM (0.546) - Negligible difference

    Hidden 256 (0.573) > 512 (0.532) - Smaller prevents overfitting

    Frozen CNN (0.569) > Fine-tuned (0.562) for BLEU-1

## 📄 Report Files

### All results are in results/ folder:

    results/*.csv - BLEU scores for all experiments

    results/*.png - Performance charts

    results/qualitative_images/ - 10 image+caption examples
