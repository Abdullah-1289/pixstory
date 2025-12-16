# PixStory Image Captioning Project

A CNN+RNN image captioning system for the PixStory case study (COSC 442).

## 📁 Project Structure

pixstory-project/
├── run_all_experiments.sh # Run all 3 experiments
├── original_implementation.py # Complete working implementation
├── requirements.txt # Python dependencies
├── config.py # Configuration
├── experiments/ # 3 separate experiments
│ ├── 01_rnn_comparison.py # LSTM vs GRU comparison
│ ├── 02_hidden_size.py # 256 vs 512 hidden units
│ └── 03_cnn_tuning.py # Frozen vs fine-tuned CNN
├── src/utils/ # Core modules
│ ├── data_loader.py # Dataset handling
│ ├── models.py # CNN encoder + RNN decoder
│ ├── trainer.py # Training loop
│ └── evaluator.py # BLEU evaluation
├── models/ # Pre-trained models (4 files)
└── results/ # Generated results
text


## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt

2. Run All Experiments
bash

chmod +x run_all_experiments.sh
./run_all_experiments.sh

3. Run Individual Experiments
bash

# Experiment 1: RNN Type Comparison
python3 experiments/01_rnn_comparison.py

# Experiment 2: Hidden Size Comparison  
python3 experiments/02_hidden_size.py

# Experiment 3: CNN Tuning Comparison
python3 experiments/03_cnn_tuning.py

🔬 Experiments
Experiment 1: RNN Type Comparison

File: experiments/01_rnn_comparison.py
Goal: Compare LSTM vs GRU decoders with frozen CNN encoder
Output: BLEU scores and comparison plot
Experiment 2: Hidden Size Comparison

File: experiments/02_hidden_size.py
Goal: Compare 256 vs 512 hidden units in RNN decoder
Output: Performance vs computation trade-off analysis
Experiment 3: CNN Tuning Comparison

File: experiments/03_cnn_tuning.py
Goal: Compare frozen vs fine-tuned CNN encoder
Output: Quality improvement vs overfitting analysis
📊 Expected Results

Each experiment generates:

    CSV file with BLEU-1 and BLEU-4 scores

    PNG plot visualizing comparisons

    Console output with qualitative examples

💾 Pre-trained Models

The models/ folder contains 4 trained models:

    exp1_gru.pth - GRU decoder, 512 hidden, frozen CNN

    exp1_lstm.pth - LSTM decoder, 512 hidden, frozen CNN

    exp2_gru_256.pth - GRU decoder, 256 hidden, frozen CNN

    exp3_finetuned.pth - GRU decoder, 512 hidden, fine-tuned CNN

Note: Experiments load these models if they exist, otherwise train new ones.
⏱️ Time Estimates

    First run: Downloads Flickr8k dataset (~1GB)

    With GPU: ~30 minutes per experiment

    With CPU: ~2 hours per experiment

    Total: 1.5-6 hours depending on hardware

✅ Verification

Test that everything works:
bash

python3 test_project.py

📝 Requirements

    Python 3.8+

    PyTorch 2.0+

    Kaggle account (dataset downloads automatically)

*Last updated: 2025-12-16*
