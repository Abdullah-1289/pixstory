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


## 📁 Project Structure

```
pixstory/
├── config.py
├── LICENSE
├── models
│   ├── exp1_gru_fixed.pth
│   ├── exp1_lstm_fixed.pth
│   ├── exp2_gru_256.pth
│   ├── exp2_gru_512.pth
│   ├── exp3_finetuned_h256_8epochs.pth
│   ├── exp3_finetuned_h256.pth
│   └── exp3_frozen_h256.pth
├── __pycache__
│   └── run_rnn_only.cpython-312.pyc
├── README.md
├── requirements.txt
├── rerun_finetuning.py
├── results
│   ├── bleu_comparison_scatter.png
│   ├── cnn_tuning_8epochs_comparison.csv
│   ├── cnn_tuning_comparison.csv
│   ├── cnn_tuning_comparison.png
│   ├── detailed_results_table.png
│   ├── experimental_results_summary.png
│   ├── final_visualizations
│   │   ├── example_01.png
│   │   ├── example_02.png
│   │   ├── example_03.png
│   │   ├── example_04.png
│   │   ├── example_05.png
│   │   ├── example_06.png
│   │   ├── example_07.png
│   │   ├── example_08.png
│   │   ├── example_09.png
│   │   └── example_10.png
│   ├── finetuned_8epochs_training.png
│   ├── hidden_size_comparison.csv
│   ├── hidden_size_comparison.png
│   ├── pixstory_recommendation.png
│   ├── qualitative_images
│   │   ├── example_01.png
│   │   ├── example_02.png
│   │   ├── example_03.png
│   │   ├── example_04.png
│   │   ├── example_05.png
│   │   ├── example_06.png
│   │   ├── example_07.png
│   │   ├── example_08.png
│   │   ├── example_09.png
│   │   └── example_10.png
│   ├── qualitative_summary_final.png
│   ├── rnn_comparison.csv
│   ├── rnn_comparison.png
│   └── training.log
├── run_all_experiments.sh
├── run_cnn_tuning.py
├── run_hidden_size.py
├── run_rnn_only.py
├── visualize_images.py
└── visualize_result.py

6 directories, 53 files

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
