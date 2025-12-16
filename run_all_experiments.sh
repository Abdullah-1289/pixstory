#!/bin/bash
# run_all_experiments.sh

echo "Starting PixStory Image Captioning Experiments"
echo "=============================================="

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "Virtual environment activated"
fi

# Create necessary directories
mkdir -p data
mkdir -p models
mkdir -p results

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt > /dev/null 2>&1

# Run all experiments
echo ""
echo "Running all experiments..."
echo "=========================="

# Run experiment 1
echo ""
echo "1. RNN Type Comparison (LSTM vs GRU)"
echo "------------------------------------"
python3 experiments/01_rnn_type_comparison.py

# Run experiment 2
echo ""
echo "2. Hidden Size Comparison (256 vs 512)"
echo "--------------------------------------"
python3 experiments/02_hidden_size_comparison.py

# Run experiment 3
echo ""
echo "3. CNN Tuning Comparison (Frozen vs Fine-tuned)"
echo "-----------------------------------------------"
python3 experiments/03_cnn_tuning_comparison.py

# Generate final report
echo ""
echo "Generating Final Report"
echo "-----------------------"
python3 run_all_experiments.py

echo ""
echo "All experiments completed!"
echo "Results saved in: results/"
echo "Models saved in: models/"