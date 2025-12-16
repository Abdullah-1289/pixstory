#!/usr/bin/env python3
"""
Experiment 1: RNN Type Comparison (LSTM vs GRU)
Requirements:
- Train at least two models: LSTM decoder and GRU decoder
- Keep other settings identical
- Compare BLEU-1 and BLEU-4 scores
- Show qualitative examples
"""
# At the top of your experiment files, copy these imports from pixstory_clean.py:
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
import nltk
nltk.download('punkt_tab', quiet=True)
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import warnings
warnings.filterwarnings('ignore')
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from src.utils import get_data_loaders, CNNEncoder, DecoderGRU, DecoderLSTM, Trainer, evaluate_model

def run_rnn_type_experiment():
    """Compare LSTM vs GRU decoders"""
    print("\n" + "="*70)
    print("EXPERIMENT 1: RNN TYPE COMPARISON (LSTM vs GRU)")
    print("="*70)
    
    # Load data
    data = get_data_loaders()
    train_loader = data['train']
    val_loader = data['val']
    test_loader = data['test']
    word2idx = data['word2idx']
    idx2word = data['idx2word']
    vocab_size = data['vocab_size']
    
    results = []
    models = {}
    
    # Test both RNN types
    rnn_types = [
        ('LSTM', DecoderLSTM),
        ('GRU', DecoderGRU)
    ]
    
    for rnn_name, DecoderClass in rnn_types:
        print(f"\n{'='*50}")
        print(f"Training {rnn_name} model")
        print(f"{'='*50}")
        
        # Create model
        encoder = CNNEncoder(EMBED_SIZE, fine_tune=False).to(DEVICE)
        decoder = DecoderClass(
            EMBED_SIZE, 
            HIDDEN_SIZE, 
            vocab_size, 
            word2idx
        ).to(DEVICE)
        
        # Check if model already exists
        model_name = f"exp_rnn_{rnn_name.lower()}"
        model_path = os.path.join(MODEL_DIR, f"{model_name}.pth")
        
        if os.path.exists(model_path):
            print(f"Loading existing {rnn_name} model...")
            encoder, decoder, word2idx = Trainer.load_model(
                model_path, encoder, decoder, word2idx, DEVICE
            )
        else:
            # Train model
            print(f"Training new {rnn_name} model...")
            trainer = Trainer(encoder, decoder, word2idx, MODEL_DIR)
            trainer.train(train_loader, val_loader, epochs=EPOCHS, 
                         fine_tune=False, save_name=model_name, device=DEVICE)
        
        # Evaluate
        evaluator = evaluate_model(encoder, decoder, idx2word, word2idx, DEVICE)
        bleu1, bleu4 = evaluate_model(encoder, decoder, test_loader, idx2word, DEVICE)
        
        print(f"\n{rnn_name} Results:")
        print(f"  BLEU-1: {bleu1:.4f}")
        print(f"  BLEU-4: {bleu4:.4f}")
        print(f"  Samples evaluated: {num_samples}")
        
        # Save results
        results.append({
            'RNN Type': rnn_name,
            'BLEU-1': bleu1,
            'BLEU-4': bleu4,
            'Hidden Size': HIDDEN_SIZE,
            'CNN Tuning': 'Frozen',
            'Parameters': sum(p.numel() for p in decoder.parameters())
        })
        
        # Save model for qualitative analysis
        models[rnn_name] = {
            'encoder': encoder,
            'decoder': decoder,
            'evaluator': evaluator
        }
    
    # Compare results
    print(f"\n{'='*70}")
    print("RNN TYPE COMPARISON RESULTS")
    print(f"{'='*70}")
    
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    
    # Save results to file
    results_file = os.path.join(RESULT_DIR, "rnn_type_comparison.csv")
    df_results.to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")
    
    # Generate qualitative comparison
    print(f"\n{'='*70}")
    print("QUALITATIVE COMPARISON")
    print(f"{'='*70}")
    
    # Get examples from each model
    for rnn_name, model_data in models.items():
        print(f"\n{rnn_name} Examples:")
        evaluator = model_data['evaluator']
        examples = evaluator.generate_examples(test_loader, num_examples=3)
        
        for i, ex in enumerate(examples):
            print(f"\n  Example {i+1}:")
            print(f"    Predicted: {ex['predicted']}")
            print(f"    True: {ex['true']}")
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # BLEU-1 comparison
    rnn_names = [r['RNN Type'] for r in results]
    bleu1_scores = [r['BLEU-1'] for r in results]
    bleu4_scores = [r['BLEU-4'] for r in results]
    
    bars1 = ax1.bar(rnn_names, bleu1_scores, color=['skyblue', 'lightcoral'])
    ax1.set_title('BLEU-1 Score Comparison')
    ax1.set_ylabel('BLEU-1 Score')
    ax1.axhline(y=0.40, color='red', linestyle='--', alpha=0.5, label='Baseline (0.40)')
    ax1.legend()
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    # BLEU-4 comparison
    bars2 = ax2.bar(rnn_names, bleu4_scores, color=['skyblue', 'lightcoral'])
    ax2.set_title('BLEU-4 Score Comparison')
    ax2.set_ylabel('BLEU-4 Score')
    ax2.axhline(y=0.10, color='red', linestyle='--', alpha=0.5, label='Baseline (0.10)')
    ax2.legend()
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plot_file = os.path.join(RESULT_DIR, "rnn_type_comparison.png")
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to: {plot_file}")
    
    # Return summary
    best_model = max(results, key=lambda x: x['BLEU-4'])
    print(f"\n{'='*70}")
    print("EXPERIMENT 1 SUMMARY")
    print(f"{'='*70}")
    print(f"Best RNN type: {best_model['RNN Type']}")
    print(f"Best BLEU-4: {best_model['BLEU-4']:.4f}")
    print(f"Improvement over baseline: {(best_model['BLEU-4'] - 0.10):.4f}")
    
    return results

if __name__ == "__main__":
    print_config()
    run_rnn_type_experiment()