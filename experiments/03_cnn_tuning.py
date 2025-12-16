#!/usr/bin/env python3
"""
Experiment 3: CNN Tuning Comparison (Frozen vs Fine-tuned)
Requirements:
- Compare frozen CNN vs fine-tuned CNN
- Evaluate BLEU-1, BLEU-4, and potential overfitting
- Discuss trade-offs in quality vs cost
"""
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from src.utils import get_data_loaders, CNNEncoder, DecoderGRU, Trainer, evaluate_model

def run_cnn_tuning_experiment():
    """Compare frozen vs fine-tuned CNN encoder"""
    print("\n" + "="*70)
    print("EXPERIMENT 3: CNN TUNING COMPARISON (Frozen vs Fine-tuned)")
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
    
    # Test both CNN tuning strategies
    tuning_strategies = [
        ('Frozen', False, EPOCHS),
        ('Fine-tuned', True, FINE_TUNE_EPOCHS)
    ]
    
    for strategy_name, fine_tune, epochs in tuning_strategies:
        print(f"\n{'='*50}")
        print(f"Training model with CNN {strategy_name.lower()}")
        print(f"{'='*50}")
        
        # Create model
        encoder = CNNEncoder(EMBED_SIZE, fine_tune=fine_tune).to(DEVICE)
        decoder = DecoderGRU(
            EMBED_SIZE, 
            HIDDEN_SIZE, 
            vocab_size, 
            word2idx
        ).to(DEVICE)
        
        # Count trainable parameters
        encoder_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        decoder_params = sum(p.numel() for p in decoder.parameters())
        total_trainable = encoder_params + decoder_params
        
        # Check if model already exists
        model_name = f"exp_cnn_{strategy_name.lower().replace('-', '_')}"
        model_path = os.path.join(MODEL_DIR, f"{model_name}.pth")
        
        if os.path.exists(model_path):
            print(f"Loading existing model with CNN {strategy_name.lower()}...")
            encoder, decoder, word2idx = Trainer.load_model(
                model_path, encoder, decoder, word2idx, DEVICE
            )
        else:
            # Train model
            print(f"Training new model with CNN {strategy_name.lower()}...")
            trainer = Trainer(encoder, decoder, word2idx, MODEL_DIR)
            trainer.train(train_loader, val_loader, epochs=epochs, 
                         fine_tune=fine_tune, save_name=model_name, device=DEVICE)
        
        # Evaluate on test set
        # Removed Evaluator instantiation
        bleu1, bleu4 = evaluate_model(encoder, decoder, test_loader, idx2word, DEVICE)
        print(f"\nCNN {strategy_name} Results:")
        print(f"  BLEU-1: {bleu1:.4f}")
        print(f"  BLEU-4: {bleu4:.4f}")
        print(f"  Trainable encoder parameters: {encoder_params:,}")
        print(f"  Total trainable parameters: {total_trainable:,}")
        print(f"  Training epochs: {epochs}")
        
        # Check for overfitting by comparing train vs test performance
        # Removed Evaluator instantiation
        train_bleu1, train_bleu4 = evaluate_model(encoder, decoder, train_loader, idx2word, DEVICE)
        overfit_score = (train_bleu4 - bleu4)  # Positive = overfitting
        print(f"  Train BLEU-4: {train_bleu4:.4f}")
        print(f"  Overfit indicator (train-test BLEU-4 diff): {overfit_score:.4f}")
        
        # Save results
        results.append({
            'CNN Tuning': strategy_name,
            'BLEU-1': bleu1,
            'BLEU-4': bleu4,
            'Train BLEU-4': train_bleu4,
            'Overfit Score': overfit_score,
            'Trainable Parameters': total_trainable,
            'Training Epochs': epochs,
            'Fine-tune': fine_tune
        })
        
        # Save model for qualitative analysis
        models[strategy_name] = {
            'encoder': encoder,
            'decoder': decoder,
            'evaluator': evaluator,
            'train_evaluator': train_evaluator
        }
    
    # Compare results
    print(f"\n{'='*70}")
    print("CNN TUNING COMPARISON RESULTS")
    print(f"{'='*70}")
    
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    
    # Save results to file
    results_file = os.path.join(RESULT_DIR, "cnn_tuning_comparison.csv")
    df_results.to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")
    
    # Generate qualitative comparison
    print(f"\n{'='*70}")
    print("QUALITATIVE COMPARISON: FROZEN vs FINE-TUNED")
    print(f"{'='*70}")
    
    # Get examples from each model
    for strategy_name, model_data in models.items():
        print(f"\nCNN {strategy_name} Examples (Test Set):")
        evaluator = model_data['evaluator']
        examples = evaluator.generate_examples(test_loader, num_examples=3)
        
        for i, ex in enumerate(examples):
            print(f"\n  Example {i+1}:")
            print(f"    Predicted: {ex['predicted']}")
            print(f"    True: {ex['true']}")
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract data
    strategies = [r['CNN Tuning'] for r in results]
    bleu1_scores = [r['BLEU-1'] for r in results]
    bleu4_scores = [r['BLEU-4'] for r in results]
    train_bleu4 = [r['Train BLEU-4'] for r in results]
    overfit_scores = [r['Overfit Score'] for r in results]
    parameters = [r['Trainable Parameters'] for r in results]
    
    # Plot 1: BLEU Scores Comparison
    x = range(len(strategies))
    width = 0.35
    
    axes[0, 0].bar(x, bleu1_scores, width, label='BLEU-1', color='skyblue')
    axes[0, 0].bar([i + width for i in x], bleu4_scores, width, label='BLEU-4', color='lightcoral')
    axes[0, 0].set_xlabel('CNN Tuning Strategy')
    axes[0, 0].set_ylabel('BLEU Score')
    axes[0, 0].set_title('Test Set Performance')
    axes[0, 0].set_xticks([i + width/2 for i in x])
    axes[0, 0].set_xticklabels(strategies)
    axes[0, 0].legend()
    axes[0, 0].axhline(y=0.40, color='red', linestyle='--', alpha=0.3, label='BLEU-1 Baseline')
    axes[0, 0].axhline(y=0.10, color='green', linestyle='--', alpha=0.3, label='BLEU-4 Baseline')
    axes[0, 0].legend()
    
    # Add value labels
    for i, (b1, b4) in enumerate(zip(bleu1_scores, bleu4_scores)):
        axes[0, 0].text(i, b1 + 0.01, f'{b1:.3f}', ha='center')
        axes[0, 0].text(i + width, b4 + 0.01, f'{b4:.3f}', ha='center')
    
    # Plot 2: Train vs Test BLEU-4 (Overfitting Analysis)
    x = np.arange(len(strategies))
    width = 0.35
    
    axes[0, 1].bar(x - width/2, train_bleu4, width, label='Train', color='lightgreen')
    axes[0, 1].bar(x + width/2, bleu4_scores, width, label='Test', color='orange')
    axes[0, 1].set_xlabel('CNN Tuning Strategy')
    axes[0, 1].set_ylabel('BLEU-4 Score')
    axes[0, 1].set_title('Train vs Test Performance (Overfitting Check)')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(strategies)
    axes[0, 1].legend()
    
    # Add value labels
    for i, (train, test) in enumerate(zip(train_bleu4, bleu4_scores)):
        axes[0, 1].text(i - width/2, train + 0.01, f'{train:.3f}', ha='center')
        axes[0, 1].text(i + width/2, test + 0.01, f'{test:.3f}', ha='center')
    
    # Plot 3: Overfitting Score
    colors = ['green' if score < 0.05 else 'orange' if score < 0.1 else 'red' 
              for score in overfit_scores]
    bars = axes[1, 0].bar(strategies, overfit_scores, color=colors)
    axes[1, 0].set_xlabel('CNN Tuning Strategy')
    axes[1, 0].set_ylabel('Overfit Score (Train BLEU-4 - Test BLEU-4)')
    axes[1, 0].set_title('Overfitting Analysis')
    axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1, 0].axhline(y=0.05, color='orange', linestyle='--', alpha=0.5, label='Mild Overfit')
    axes[1, 0].axhline(y=0.10, color='red', linestyle='--', alpha=0.5, label='Severe Overfit')
    axes[1, 0].legend()
    
    # Add value labels and annotations
    for i, (bar, score) in enumerate(zip(bars, overfit_scores)):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, 
                       bar.get_height() + (0.01 if bar.get_height() >= 0 else -0.02),
                       f'{score:.3f}', ha='center')
        
        if score < 0.05:
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                           'Good', ha='center', va='center', color='white', fontweight='bold')
        elif score < 0.1:
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                           'Mild', ha='center', va='center', color='white', fontweight='bold')
        else:
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                           'Overfit', ha='center', va='center', color='white', fontweight='bold')
    
    # Plot 4: Parameters vs Performance Trade-off
    scatter = axes[1, 1].scatter([p/1e6 for p in parameters], bleu4_scores, 
                                s=200, c=overfit_scores, cmap='RdYlGn_r')
    axes[1, 1].set_xlabel('Trainable Parameters (millions)')
    axes[1, 1].set_ylabel('Test BLEU-4 Score')
    axes[1, 1].set_title('Efficiency vs Performance Trade-off')
    
    # Add strategy labels
    for i, (param, bleu4, strategy) in enumerate(zip(parameters, bleu4_scores, strategies)):
        axes[1, 1].annotate(strategy, (param/1e6, bleu4), 
                           xytext=(5, 5), textcoords='offset points',
                           fontweight='bold')
    
    # Add colorbar for overfit score
    cbar = plt.colorbar(scatter, ax=axes[1, 1])
    cbar.set_label('Overfit Score (lower is better)')
    
    plt.tight_layout()
    plot_file = os.path.join(RESULT_DIR, "cnn_tuning_comparison.png")
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to: {plot_file}")
    
    # Return summary
    best_model = max(results, key=lambda x: x['BLEU-4'])
    print(f"\n{'='*70}")
    print("EXPERIMENT 3 SUMMARY")
    print(f"{'='*70}")
    print(f"Best CNN tuning strategy: {best_model['CNN Tuning']}")
    print(f"Best BLEU-4: {best_model['BLEU-4']:.4f}")
    print(f"Overfit score: {best_model['Overfit Score']:.4f}")
    print(f"Trainable parameters: {best_model['Trainable Parameters']:,}")
    
    # Recommendation based on trade-offs
    frozen = results[0] if results[0]['CNN Tuning'] == 'Frozen' else results[1]
    finetuned = results[1] if results[1]['CNN Tuning'] == 'Fine-tuned' else results[0]
    
    improvement = finetuned['BLEU-4'] - frozen['BLEU-4']
    param_increase = finetuned['Trainable Parameters'] / frozen['Trainable Parameters']
    
    print(f"\nTrade-off Analysis:")
    print(f"  BLEU-4 improvement from fine-tuning: {improvement:.4f}")
    print(f"  Parameter increase: {param_increase:.2f}x")
    print(f"  Overfit increase: {finetuned['Overfit Score'] - frozen['Overfit Score']:.4f}")
    
    if improvement > 0.03 and finetuned['Overfit Score'] < 0.08:
        print("  → RECOMMENDATION: Fine-tune CNN for better performance")
        print("    (Good improvement with acceptable overfitting)")
    elif improvement > 0.05:
        print("  → RECOMMENDATION: Fine-tune CNN (significant improvement)")
        print("    (Consider regularization if overfitting is high)")
    else:
        print("  → RECOMMENDATION: Use frozen CNN for efficiency")
        print("    (Minimal improvement doesn't justify extra parameters)")
    
    return results

if __name__ == "__main__":
    print_config()
    run_cnn_tuning_experiment()