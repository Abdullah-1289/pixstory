#!/usr/bin/env python3
"""
Experiment 2: Hidden Size Comparison (256 vs 512)
Requirements:
- Train at least two configurations (hidden size = 256 vs 512)
- Document BLEU-1 and BLEU-4 scores
- Comment on training time, memory usage, and qualitative differences
"""
import torch
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from src.utils import get_data_loaders, CNNEncoder, DecoderGRU, Trainer, evaluate_model

def run_hidden_size_experiment():
    """Compare different hidden sizes"""
    print("\n" + "="*70)
    print("EXPERIMENT 2: HIDDEN SIZE COMPARISON (256 vs 512)")
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
    
    # Test different hidden sizes
    hidden_sizes = [
        (256, 'Small'),
        (512, 'Large')
    ]
    
    for hidden_size, size_name in hidden_sizes:
        print(f"\n{'='*50}")
        print(f"Training model with hidden size = {hidden_size}")
        print(f"{'='*50}")
        
        # Create model
        encoder = CNNEncoder(EMBED_SIZE, fine_tune=False).to(DEVICE)
        decoder = DecoderGRU(
            EMBED_SIZE, 
            hidden_size, 
            vocab_size, 
            word2idx
        ).to(DEVICE)
        
        # Count parameters
        decoder_params = sum(p.numel() for p in decoder.parameters())
        total_params = decoder_params + sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        
        # Check if model already exists
        model_name = f"exp_hidden_{hidden_size}"
        model_path = os.path.join(MODEL_DIR, f"{model_name}.pth")
        
        start_time = time.time()
        
        if os.path.exists(model_path):
            print(f"Loading existing model with hidden size {hidden_size}...")
            encoder, decoder, word2idx = Trainer.load_model(
                model_path, encoder, decoder, word2idx, DEVICE
            )
            training_time = 0  # Already trained
        else:
            # Train model and measure time
            print(f"Training new model with hidden size {hidden_size}...")
            trainer = Trainer(encoder, decoder, word2idx, MODEL_DIR)
            training_time = time.time()
            trainer.train(train_loader, val_loader, epochs=EPOCHS, 
                         fine_tune=False, save_name=model_name, device=DEVICE)
            training_time = time.time() - training_time
        
        # Evaluate
        # Removed Evaluator instantiation
        bleu1, bleu4 = evaluate_model(encoder, decoder, test_loader, idx2word, DEVICE)
        print(f"\nHidden Size {hidden_size} Results:")
        print(f"  BLEU-1: {bleu1:.4f}")
        print(f"  BLEU-4: {bleu4:.4f}")
        print(f"  Decoder parameters: {decoder_params:,}")
        print(f"  Trainable parameters: {total_params:,}")
        print(f"  Training time: {training_time:.1f} seconds")
        
        # Memory usage estimate
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.max_memory_allocated() / 1024**2  # MB
            print(f"  GPU memory used: {memory_allocated:.1f} MB")
            torch.cuda.reset_peak_memory_stats()
        
        # Save results
        results.append({
            'Hidden Size': hidden_size,
            'Size Name': size_name,
            'BLEU-1': bleu1,
            'BLEU-4': bleu4,
            'Training Time (s)': training_time,
            'Decoder Parameters': decoder_params,
            'Trainable Parameters': total_params
        })
        
        # Save model for qualitative analysis
        models[hidden_size] = {
            'encoder': encoder,
            'decoder': decoder,
            'evaluator': evaluator
        }
    
    # Compare results
    print(f"\n{'='*70}")
    print("HIDDEN SIZE COMPARISON RESULTS")
    print(f"{'='*70}")
    
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    
    # Save results to file
    results_file = os.path.join(RESULT_DIR, "hidden_size_comparison.csv")
    df_results.to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")
    
    # Generate qualitative comparison
    print(f"\n{'='*70}")
    print("QUALITATIVE COMPARISON BY HIDDEN SIZE")
    print(f"{'='*70}")
    
    # Get examples from each model
    for hidden_size, model_data in models.items():
        print(f"\nHidden Size {hidden_size} Examples:")
        evaluator = model_data['evaluator']
        examples = evaluator.generate_examples(test_loader, num_examples=3)
        
        for i, ex in enumerate(examples):
            print(f"\n  Example {i+1}:")
            print(f"    Predicted: {ex['predicted']}")
            print(f"    True: {ex['true']}")
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # BLEU scores comparison
    sizes = [r['Hidden Size'] for r in results]
    size_names = [r['Size Name'] for r in results]
    bleu1_scores = [r['BLEU-1'] for r in results]
    bleu4_scores = [r['BLEU-4'] for r in results]
    training_times = [r['Training Time (s)'] for r in results]
    parameters = [r['Decoder Parameters'] for r in results]
    
    # Plot 1: BLEU Scores
    x = range(len(sizes))
    width = 0.35
    
    axes[0, 0].bar(x, bleu1_scores, width, label='BLEU-1', color='skyblue')
    axes[0, 0].bar([i + width for i in x], bleu4_scores, width, label='BLEU-4', color='lightcoral')
    axes[0, 0].set_xlabel('Hidden Size')
    axes[0, 0].set_ylabel('BLEU Score')
    axes[0, 0].set_title('BLEU Score Comparison')
    axes[0, 0].set_xticks([i + width/2 for i in x])
    axes[0, 0].set_xticklabels(sizes)
    axes[0, 0].legend()
    axes[0, 0].axhline(y=0.40, color='red', linestyle='--', alpha=0.3)
    axes[0, 0].axhline(y=0.10, color='green', linestyle='--', alpha=0.3)
    
    # Add value labels
    for i, (b1, b4) in enumerate(zip(bleu1_scores, bleu4_scores)):
        axes[0, 0].text(i, b1 + 0.01, f'{b1:.3f}', ha='center')
        axes[0, 0].text(i + width, b4 + 0.01, f'{b4:.3f}', ha='center')
    
    # Plot 2: Training Time
    axes[0, 1].bar(size_names, training_times, color=['lightgreen', 'orange'])
    axes[0, 1].set_xlabel('Hidden Size')
    axes[0, 1].set_ylabel('Training Time (seconds)')
    axes[0, 1].set_title('Training Time Comparison')
    
    for i, time_val in enumerate(training_times):
        axes[0, 1].text(i, time_val + max(training_times)*0.02, 
                       f'{time_val:.1f}s', ha='center')
    
    # Plot 3: Parameters
    axes[1, 0].bar(size_names, [p/1e6 for p in parameters], 
                   color=['lightblue', 'lightpink'])
    axes[1, 0].set_xlabel('Hidden Size')
    axes[1, 0].set_ylabel('Parameters (millions)')
    axes[1, 0].set_title('Model Size Comparison (Decoder Only)')
    
    for i, param_count in enumerate(parameters):
        axes[1, 0].text(i, (param_count/1e6) + 0.1, 
                       f'{param_count/1e6:.2f}M', ha='center')
    
    # Plot 4: Trade-off analysis
    axes[1, 1].scatter([p/1e6 for p in parameters], bleu4_scores, 
                      s=[t/10 for t in training_times], c=sizes, cmap='viridis')
    axes[1, 1].set_xlabel('Parameters (millions)')
    axes[1, 1].set_ylabel('BLEU-4 Score')
    axes[1, 1].set_title('Trade-off: Parameters vs Performance')
    
    # Add annotations
    for i, (param, bleu4, size) in enumerate(zip(parameters, bleu4_scores, sizes)):
        axes[1, 1].annotate(f'{size}', (param/1e6, bleu4), 
                           xytext=(5, 5), textcoords='offset points')
    
    plt.tight_layout()
    plot_file = os.path.join(RESULT_DIR, "hidden_size_comparison.png")
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to: {plot_file}")
    
    # Return summary
    best_model = max(results, key=lambda x: x['BLEU-4'])
    print(f"\n{'='*70}")
    print("EXPERIMENT 2 SUMMARY")
    print(f"{'='*70}")
    print(f"Best hidden size: {best_model['Hidden Size']}")
    print(f"Best BLEU-4: {best_model['BLEU-4']:.4f}")
    print(f"Parameters: {best_model['Decoder Parameters']:,}")
    print(f"Training time: {best_model['Training Time (s)']:.1f}s")
    
    # Recommendation
    small_model = results[0]  # 256
    large_model = results[1]  # 512
    
    improvement = large_model['BLEU-4'] - small_model['BLEU-4']
    param_increase = large_model['Decoder Parameters'] / small_model['Decoder Parameters']
    time_increase = large_model['Training Time (s)'] / small_model['Training Time (s)'] if small_model['Training Time (s)'] > 0 else 1
    
    print(f"\nRecommendation Analysis:")
    print(f"  BLEU-4 improvement: {improvement:.4f}")
    print(f"  Parameter increase: {param_increase:.2f}x")
    print(f"  Training time increase: {time_increase:.2f}x")
    
    if improvement > 0.02:  # Significant improvement
        print("  → RECOMMENDATION: Use larger hidden size (512) for better performance")
    else:
        print("  → RECOMMENDATION: Use smaller hidden size (256) for efficiency")
    
    return results

if __name__ == "__main__":
    print_config()
    run_hidden_size_experiment()