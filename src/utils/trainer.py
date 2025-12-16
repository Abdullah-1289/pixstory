"""
Training utilities
"""
import torch
import torch.nn as nn
from tqdm import tqdm
import os

def train_epoch(encoder, decoder, train_loader, criterion, optimizer, vocab_size, device):
    encoder.train()
    decoder.train()
    train_loss = 0
    
    for images, captions in tqdm(train_loader, desc="Training"):
        images, captions = images.to(device), captions.to(device)
        features = encoder(images)
        outputs = decoder(features, captions[:, :-1])
        loss = criterion(outputs.reshape(-1, vocab_size), captions[:, 1:].reshape(-1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters() + decoder.parameters(), 5.0)
        optimizer.step()
        train_loss += loss.item()
    
    return train_loss / len(train_loader)

def validate(encoder, decoder, val_loader, criterion, vocab_size, device):
    encoder.eval()
    decoder.eval()
    val_loss = 0
    
    with torch.no_grad():
        for images, captions in val_loader:
            images, captions = images.to(device), captions.to(device)
            features = encoder(images)
            outputs = decoder(features, captions[:, :-1])
            loss = criterion(outputs.reshape(-1, vocab_size), captions[:, 1:].reshape(-1))
            val_loss += loss.item()
    
    return val_loss / len(val_loader)

def save_model(encoder, decoder, word2idx, vocab_size, path):
    torch.save({
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'word2idx': word2idx,
        'vocab_size': vocab_size
    }, path)
