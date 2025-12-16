# src/utils/trainer.py
import torch
import torch.nn as nn
from tqdm import tqdm
import os

class Trainer:
    def __init__(self, encoder, decoder, word2idx, save_dir="models"):
        self.encoder = encoder
        self.decoder = decoder
        self.word2idx = word2idx
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def train(self, train_loader, val_loader, epochs=8, fine_tune=False, save_name="model", device='cuda'):
        """Train the model"""
        
        # Setup optimizer
        if fine_tune:
            params = list(self.decoder.parameters()) + list(self.encoder.parameters())
            lr = 1e-4
        else:
            params = self.decoder.parameters()
            lr = 1e-3
        
        optimizer = torch.optim.Adam(params, lr=lr)
        criterion = nn.CrossEntropyLoss(ignore_index=self.word2idx['<pad>'])
        
        self.encoder.train()
        self.decoder.train()
        self.encoder.to(device)
        self.decoder.to(device)
        
        for epoch in range(epochs):
            train_loss = 0
            self.encoder.train()
            self.decoder.train()
            
            for images, captions in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                images, captions = images.to(device), captions.to(device)
                
                # Forward pass
                features = self.encoder(images)
                outputs = self.decoder(features, captions[:, :-1])
                
                # Calculate loss
                loss = criterion(
                    outputs.reshape(-1, outputs.size(-1)),
                    captions[:, 1:].reshape(-1)
                )
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.decoder.parameters()), 5.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")
        
        # Save model
        self.save_model(save_name)
        return avg_train_loss
    
    def save_model(self, name):
        """Save model checkpoint"""
        checkpoint = {
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'word2idx': self.word2idx,
        }
        
        path = os.path.join(self.save_dir, f"{name}.pth")
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
    
@staticmethod
def load_model(path, encoder, decoder, word2idx=None, device='cuda'):
    """Load model checkpoint - handles different formats"""
    checkpoint = torch.load(path, map_location=device)
    
    # Try different key patterns
    if 'encoder_state_dict' in checkpoint and 'decoder_state_dict' in checkpoint:
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
    elif 'encoder' in checkpoint and 'decoder' in checkpoint:
        # YOUR FORMAT - state_dicts are in 'encoder' and 'decoder' keys
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
    else:
        # Try direct loading
        encoder.load_state_dict(checkpoint)
    
    # Get word2idx if available
    if word2idx is None:
        word2idx = checkpoint.get('word2idx', checkpoint.get('word2idx', {}))
    
    print(f"Model loaded from {path}")
    return encoder, decoder, word2idx
