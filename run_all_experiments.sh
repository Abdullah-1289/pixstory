#!/bin/bash
# run_all_experiments.sh - Clean version

echo "=== PIXSTORY EXPERIMENTS ==="
echo "Running all experiments..."

# 1. RNN Type Comparison
echo "1. RNN Type Comparison (GRU vs LSTM)..."
python3 run_rnn_only.py

# 2. Hidden Size Comparison  
echo "2. Hidden Size Comparison (256 vs 512)..."
python3 run_hidden_size.py

# 3. CNN Tuning Comparison
echo "3. CNN Tuning Comparison (Frozen vs Fine-tuned)..."
python3 run_cnn_tuning.py

# 4. 8-epoch Fine-tuning Test
echo "4. 8-epoch Fine-tuning Test..."
python3 rerun_finetuning.py

echo "=== EXPERIMENTS COMPLETE ==="
echo "Results saved to: results/"
echo ""
echo "To generate visualizations:"
echo "  python3 visualize_images.py    # Qualitative examples"
echo "  python3 visualize_result.py    # Performance charts"
