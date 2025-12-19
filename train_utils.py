import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

def train_with_validation(encoder, decoder, train_loader, val_loader, 
                          epochs=10, fine_tune=False, model_name="model"):
    """
    Training function with validation tracking and early stopping.
    """
    device = next(encoder.parameters()).device
    
    # Setup optimizer
    if fine_tune:
        params = list(decoder.parameters()) + list(encoder.parameters())
        lr = 1e-4
    else:
        params = decoder.parameters()
        lr = 1e-3
    
    optimizer = torch.optim.Adam(params, lr=lr)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)  # <pad>=0
    
    # Tracking
    train_losses = []
    val_losses = []
    val_bleu1_scores = []
    val_bleu4_scores = []
    best_bleu1 = 0
    patience = 3
    patience_counter = 0
    
    os.makedirs("checkpoints", exist_ok=True)
    
    for epoch in range(epochs):
        # Training
        encoder.train()
        decoder.train()
        epoch_train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, captions in pbar:
            images, captions = images.to(device), captions.to(device)
            
            # Forward pass
            features = encoder(images)
            outputs = decoder(features, captions[:, :-1])
            
            # Calculate loss
            loss = criterion(
                outputs.reshape(-1, outputs.size(-1)),
                captions[:, 1:].reshape(-1)
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            optimizer.step()
            
            epoch_train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        encoder.eval()
        decoder.eval()
        epoch_val_loss = 0
        bleu1_scores = []
        bleu4_scores = []
        
        with torch.no_grad():
            for images, captions in val_loader:
                images, captions = images.to(device), captions.to(device)
                
                # Calculate validation loss
                features = encoder(images)
                outputs = decoder(features, captions[:, :-1])
                loss = criterion(
                    outputs.reshape(-1, outputs.size(-1)),
                    captions[:, 1:].reshape(-1)
                )
                epoch_val_loss += loss.item()
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Calculate validation BLEU (you need to implement this)
        # val_bleu1, val_bleu4 = evaluate_on_validation(encoder, decoder, val_loader)
        # val_bleu1_scores.append(val_bleu1)
        # val_bleu4_scores.append(val_bleu4)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Early stopping check (commented until BLEU is implemented)
        # if val_bleu1 > best_bleu1:
        #     best_bleu1 = val_bleu1
        #     patience_counter = 0
        #     # Save best model
        #     torch.save({
        #         'epoch': epoch,
        #         'encoder_state': encoder.state_dict(),
        #         'decoder_state': decoder.state_dict(),
        #         'optimizer_state': optimizer.state_dict(),
        #         'train_loss': avg_train_loss,
        #         'val_loss': avg_val_loss,
        #         'val_bleu1': val_bleu1,
        #         'val_bleu4': val_bleu4,
        #     }, f"checkpoints/{model_name}_best.pth")
        # else:
        #     patience_counter += 1
        #     if patience_counter >= patience:
        #         print(f"Early stopping at epoch {epoch+1}")
        #         break
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # plt.subplot(1, 2, 2)
    # plt.plot(val_bleu1_scores, label='BLEU-1')
    # plt.plot(val_bleu4_scores, label='BLEU-4')
    # plt.xlabel('Epoch')
    # plt.ylabel('BLEU Score')
    # plt.legend()
    # plt.title('Validation BLEU Scores')
    
    plt.tight_layout()
    plt.savefig(f"checkpoints/{model_name}_curves.png", dpi=150)
    plt.close()
    
    return encoder, decoder, train_losses, val_losses

