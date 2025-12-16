#!/bin/bash
echo "Running all PixStory experiments..."

echo "=== Experiment 1: RNN Type Comparison ==="
python experiments/01_rnn_type_comparison.py

echo "=== Experiment 2: Hidden Size Comparison ==="
# This would train GRU-256
echo "Skipping - use existing model: models/exp2_gru_256.pth"

echo "=== Experiment 3: CNN Tuning Comparison ==="  
echo "Skipping - use existing model: models/exp3_finetuned.pth"

echo "✅ All experiments referenced. Models available in models/"
