#!/usr/bin/env python3
"""
Experiment 1: Compare GRU vs LSTM
"""
import sys
sys.path.append('src')

import torch
import torch.nn as nn
import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split

from utils.data_loader import build_vocab, FlickrDataset, get_transforms
from utils.models import CNNEncoder, DecoderGRU, DecoderLSTM
from utils.trainer import train_epoch, validate, save_model
from utils.evaluator import evaluate_model
from torch.utils.data import DataLoader

def main():
    print("=== Experiment 1: RNN Type Comparison (GRU vs LSTM) ===\n")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    epochs = 8
    embed_size = 512
    hidden_size = 512
    
    # Load data
    print("1. Loading dataset...")
    path = kagglehub.dataset_download("adityajn105/flickr8k")
    df = pd.read_csv(f"{path}/captions.txt")
    
    # Split data
    unique_images = df['image'].unique()
    train_imgs, temp = train_test_split(unique_images, test_size=0.2, random_state=42)
    val_imgs, test_imgs = train_test_split(temp, test_size=0.5, random_state=42)
    
    train_df = df[df['image'].isin(train_imgs)]
    val_df = df[df['image'].isin(val_imgs)]
    test_df = df[df['image'].isin(test_imgs)]
    
    # Build vocabulary
    print("2. Building vocabulary...")
    word2idx, idx2word, vocab_size = build_vocab(train_df)
    
    # Create datasets
    transform = get_transforms()
    train_dataset = FlickrDataset(train_df, word2idx, f"{path}/Images", transform=transform)
    val_dataset = FlickrDataset(val_df, word2idx, f"{path}/Images", transform=transform)
    test_dataset = FlickrDataset(test_df, word2idx, f"{path}/Images", transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    criterion = nn.CrossEntropyLoss(ignore_index=word2idx['<pad>'])
    
    # Train GRU model
    print("\n3. Training GRU model...")
    encoder_gru = CNNEncoder(embed_size, fine_tune=False).to(device)
    decoder_gru = DecoderGRU(embed_size, hidden_size, vocab_size, word2idx).to(device)
    
    optimizer = torch.optim.Adam(decoder_gru.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        train_loss = train_epoch(encoder_gru, decoder_gru, train_loader, criterion, optimizer, vocab_size, device)
        val_loss = validate(encoder_gru, decoder_gru, val_loader, criterion, vocab_size, device)
        print(f"   Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Save GRU model
    save_model(encoder_gru, decoder_gru, word2idx, vocab_size, "models/exp1_gru.pth")
    
    # Evaluate GRU
    bleu1_gru, bleu4_gru = evaluate_model(encoder_gru, decoder_gru, test_loader, idx2word, device)
    print(f"   GRU BLEU-1: {bleu1_gru:.4f}, BLEU-4: {bleu4_gru:.4f}")
    
    # Train LSTM model
    print("\n4. Training LSTM model...")
    encoder_lstm = CNNEncoder(embed_size, fine_tune=False).to(device)
    decoder_lstm = DecoderLSTM(embed_size, hidden_size, vocab_size, word2idx).to(device)
    
    optimizer = torch.optim.Adam(decoder_lstm.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        train_loss = train_epoch(encoder_lstm, decoder_lstm, train_loader, criterion, optimizer, vocab_size, device)
        val_loss = validate(encoder_lstm, decoder_lstm, val_loader, criterion, vocab_size, device)
        print(f"   Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Save LSTM model
    save_model(encoder_lstm, decoder_lstm, word2idx, vocab_size, "models/exp1_lstm.pth")
    
    # Evaluate LSTM
    bleu1_lstm, bleu4_lstm = evaluate_model(encoder_lstm, decoder_lstm, test_loader, idx2word, device)
    print(f"   LSTM BLEU-1: {bleu1_lstm:.4f}, BLEU-4: {bleu4_lstm:.4f}")
    
    # Save results
    results = pd.DataFrame({
        'Model': ['GRU-512', 'LSTM-512'],
        'BLEU-1': [bleu1_gru, bleu1_lstm],
        'BLEU-4': [bleu4_gru, bleu4_lstm]
    })
    results.to_csv("results/experiment1_results.csv", index=False)
    
    print("\n✅ Experiment 1 completed!")
    print(results.to_markdown())

if __name__ == '__main__':
    main()
