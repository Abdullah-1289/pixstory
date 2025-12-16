"""
Evaluation utilities
"""
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch

def evaluate_model(encoder, decoder, test_loader, idx2word, device='cuda'):
    encoder.eval()
    decoder.eval()
    bleu1_scores, bleu4_scores = [], []
    
    with torch.no_grad():
        for images, captions in test_loader:
            images = images.to(device)
            features = encoder(images)
            pred_ids = decoder.predict(features, device=device)
            
            for i in range(len(pred_ids)):
                pred_tokens = []
                for idx in pred_ids[i]:
                    word = idx2word[idx.item()]
                    if word in ['<start>', '<end>', '<pad>']:
                        continue
                    pred_tokens.append(word)
                
                true_tokens = []
                for idx in captions[i][1:]:
                    word = idx2word[idx.item()]
                    if word in ['<start>', '<end>', '<pad>']:
                        continue
                    true_tokens.append(word)
                
                if pred_tokens and true_tokens:
                    bleu1 = sentence_bleu([true_tokens], pred_tokens, weights=(1, 0, 0, 0))
                    bleu4 = sentence_bleu([true_tokens], pred_tokens,
                                         smoothing_function=SmoothingFunction().method1)
                    bleu1_scores.append(bleu1)
                    bleu4_scores.append(bleu4)
    
    return np.mean(bleu1_scores), np.mean(bleu4_scores)
