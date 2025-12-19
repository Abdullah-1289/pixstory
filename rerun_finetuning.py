#!/usr/bin/env python3
"""Re-run Fine-tuning Only (8 epochs)"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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

# ============ SETUP ============
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Download dataset
import kagglehub
print("Loading dataset...")
path = kagglehub.dataset_download("adityajn105/flickr8k")
image_dir = f"{path}/Images"
captions_file = f"{path}/captions.txt"

# Load data (MUST use same splits as previous experiments)
df = pd.read_csv(captions_file)
print(f"Total captions: {len(df)}")
unique_images = df['image'].unique()
train_imgs, temp_imgs = train_test_split(unique_images, test_size=0.2, random_state=42)
val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

# Build vocabulary with min_freq=2 (MUST match previous experiments)
print("Building vocabulary (min_freq=2)...")
train_df = df[df['image'].isin(train_imgs)]
all_tokens = []
for caption in train_df['caption']:
    tokens = nltk.word_tokenize(caption.lower())
    all_tokens.extend(tokens)

word_counts = Counter(all_tokens)
vocab = ['<pad>', '<start>', '<end>', '<unk>']
vocab += [word for word, count in word_counts.items() if count >= 2]

word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for word, idx in word2idx.items()}
vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")

# Dataset class (same as before)
class FlickrDataset(Dataset):
    def __init__(self, df, word2idx, image_dir, max_len=25, transform=None):
        self.df = df; self.word2idx = word2idx; self.image_dir = image_dir
        self.max_len = max_len; self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(f"{self.image_dir}/{row['image']}").convert('RGB')
        if self.transform: image = self.transform(image)
        tokens = nltk.word_tokenize(row['caption'].lower())
        tokens = ['<start>'] + tokens + ['<end>']
        tokens = tokens[:self.max_len] + ['<pad>'] * (self.max_len - len(tokens))
        caption_ids = [self.word2idx.get(t, word2idx['<unk>']) for t in tokens]
        return image, torch.tensor(caption_ids)

transform = T.Compose([T.Resize((224, 224)), T.ToTensor(), 
                       T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
train_dataset = FlickrDataset(df[df['image'].isin(train_imgs)], word2idx, image_dir, transform=transform)
val_dataset = FlickrDataset(df[df['image'].isin(val_imgs)], word2idx, image_dir, transform=transform)
test_dataset = FlickrDataset(df[df['image'].isin(test_imgs)], word2idx, image_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
print(f"Data loaded: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")

# ============ MODELS ============
class CNNEncoder(nn.Module):
    def __init__(self, embed_size, fine_tune=False):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Fine-tuning: last 30 layers trainable
        if fine_tune:
            for param in list(self.features.parameters())[-30:]:
                param.requires_grad = True
            print("  CNN encoder: Fine-tuning enabled (last 30 layers)")
        else:
            for param in self.features.parameters():
                param.requires_grad = False
            print("  CNN encoder: Frozen (feature extractor)")
        
        self.projection = nn.Sequential(
            nn.Linear(resnet.fc.in_features, embed_size),
            nn.BatchNorm1d(embed_size),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.projection(x)

class DecoderGRU(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, word2idx, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.word2idx = word2idx
        self.vocab_size = vocab_size
        self.feature_projection = nn.Linear(embed_size, hidden_size) if embed_size != hidden_size else nn.Identity()

    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        hidden = self.feature_projection(features).unsqueeze(0)
        output, _ = self.gru(embeddings, hidden)
        output = self.dropout(output)
        return self.linear(output)

    def predict(self, features, max_len=25, device='cuda'):
        batch_size = features.size(0)
        hidden = self.feature_projection(features).unsqueeze(0)
        inputs = torch.tensor([[self.word2idx['<start>']]] * batch_size, device=device)
        seq = []
        for _ in range(max_len):
            embeddings = self.embed(inputs)
            output, hidden = self.gru(embeddings, hidden)
            predicted = self.linear(output.squeeze(1)).argmax(1)
            seq.append(predicted.unsqueeze(1))
            inputs = predicted.unsqueeze(1)
            if (predicted == self.word2idx['<end>']).all(): break
        return torch.cat(seq, 1)

# ============ TRAINING ============
def train_model(encoder, decoder, train_loader, val_loader, epochs=8, fine_tune=False, save_name="model"):
    if fine_tune:
        params = list(decoder.parameters()) + list(encoder.parameters())
        lr = 1e-4
        print(f"  Training with fine-tuning (LR: {lr}) for {epochs} epochs")
    else:
        params = decoder.parameters()
        lr = 1e-3
        print(f"  Training decoder only (LR: {lr}) for {epochs} epochs")
    
    optimizer = torch.optim.Adam(params, lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=word2idx['<pad>'])
    
    os.makedirs("models", exist_ok=True)
    
    train_losses = []
    
    for epoch in range(epochs):
        encoder.train(); decoder.train()
        train_loss = 0
        for images, captions in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, captions = images.to(device), captions.to(device)
            features = encoder(images)
            outputs = decoder(features, captions[:, :-1])
            loss = criterion(outputs.reshape(-1, vocab_size), captions[:, 1:].reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            optimizer.step()
            train_loss += loss.item()
        
        avg_train = train_loss / len(train_loader)
        train_losses.append(avg_train)
        print(f"  Epoch {epoch+1}: Train Loss: {avg_train:.4f}")
    
    torch.save({
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'word2idx': word2idx,
        'idx2word': idx2word,
        'vocab_size': vocab_size,
        'hidden_size': decoder.gru.hidden_size,
        'fine_tuned': fine_tune
    }, f"models/{save_name}.pth")
    print(f"  Model saved: models/{save_name}.pth")
    
    return encoder, decoder, train_losses

# ============ MULTI-REFERENCE EVALUATION ============
def evaluate_multi_ref(encoder, decoder, test_df, image_dir, transform, word2idx, idx2word, device='cuda'):
    encoder.eval(); decoder.eval()
    
    # Group ALL references by image
    image_to_refs = {}
    for _, row in test_df.iterrows():
        img_name = row['image']
        tokens = nltk.word_tokenize(row['caption'].lower())
        image_to_refs.setdefault(img_name, []).append(tokens)
    
    bleu1_scores = []; bleu4_scores = []
    smoothie = SmoothingFunction().method1
    
    with torch.no_grad():
        for img_name, refs_list in tqdm(image_to_refs.items(), desc="Evaluating"):
            img_path = f"{image_dir}/{img_name}"
            image = Image.open(img_path).convert('RGB')
            image_t = transform(image).unsqueeze(0).to(device)
            
            features = encoder(image_t)
            pred_ids = decoder.predict(features, device=device)
            
            pred_tokens = [idx2word[idx.item()] for idx in pred_ids[0] 
                          if idx2word[idx.item()] not in ['<start>','<end>','<pad>','<unk>']]
            
            if pred_tokens and len(refs_list) > 0:
                bleu1 = sentence_bleu(refs_list[:2], pred_tokens, weights=(1, 0, 0, 0))  # Using 2 refs
                bleu4 = sentence_bleu(refs_list[:2], pred_tokens, smoothing_function=smoothie)  # Using 2 refs
                bleu1_scores.append(bleu1); bleu4_scores.append(bleu4)
    
    return np.mean(bleu1_scores), np.mean(bleu4_scores)

# ============ RE-RUN FINE-TUNING ONLY ============
print("\n" + "="*70)
print("RE-RUNNING FINE-TUNED CNN (8 EPOCHS - FAIR COMPARISON)")
print("="*70)

os.makedirs("results", exist_ok=True)
test_df = df[df['image'].isin(test_imgs)].copy()

HIDDEN_SIZE = 256
EMBED_SIZE = 512
print(f"Using hidden_size={HIDDEN_SIZE}")

# Delete previous fine-tuned model
model_name = f"exp3_finetuned_h{HIDDEN_SIZE}_8epochs"
model_path = f"models/{model_name}.pth"
print(f"Model will be saved as: {model_name}")

# Load your previous frozen results for comparison
frozen_results = {'BLEU-1': 0.5688, 'BLEU-4': 0.1439}
print(f"\nPrevious Frozen CNN (8 epochs):")
print(f"  BLEU-1: {frozen_results['BLEU-1']:.4f}, BLEU-4: {frozen_results['BLEU-4']:.4f}")

# Train fine-tuned model for 8 epochs
print(f"\n{'='*50}")
print("FINE-TUNED CNN (8 epochs - same as frozen)")
print(f"{'='*50}")

encoder = CNNEncoder(EMBED_SIZE, fine_tune=True).to(device)
decoder = DecoderGRU(EMBED_SIZE, HIDDEN_SIZE, vocab_size, word2idx).to(device)

encoder, decoder, train_losses = train_model(
    encoder, decoder, train_loader, val_loader,
    epochs=8, fine_tune=True, save_name=model_name
)

# Evaluate fine-tuned model
bleu1_finetuned, bleu4_finetuned = evaluate_multi_ref(
    encoder, decoder, test_df, image_dir,
    transform, word2idx, idx2word, device
)

print(f"\nFine-tuned CNN Results (8 epochs):")
print(f"  BLEU-1: {bleu1_finetuned:.4f}")
print(f"  BLEU-4: {bleu4_finetuned:.4f}")

# Compare with frozen
improvement_bleu1 = bleu1_finetuned - frozen_results['BLEU-1']
improvement_bleu4 = bleu4_finetuned - frozen_results['BLEU-4']

print("\n" + "="*70)
print("COMPARISON: 8 EPOCHS EACH")
print("="*70)
print(f"{'Model':<15} {'BLEU-1':<10} {'BLEU-4':<10}")
print(f"{'-'*35}")
print(f"{'Frozen CNN':<15} {frozen_results['BLEU-1']:<10.4f} {frozen_results['BLEU-4']:<10.4f}")
print(f"{'Fine-tuned CNN':<15} {bleu1_finetuned:<10.4f} {bleu4_finetuned:<10.4f}")
print(f"{'Improvement':<15} {improvement_bleu1:<+10.4f} {improvement_bleu4:<+10.4f}")
print(f"{'% Change':<15} {improvement_bleu1/frozen_results['BLEU-1']*100:<+10.1f}% {improvement_bleu4/frozen_results['BLEU-4']*100:<+10.1f}%")

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(train_losses, 's-', linewidth=2, markersize=8, label='Fine-tuned CNN')
plt.axhline(y=2.4908, color='skyblue', linestyle='--', linewidth=2, label='Frozen CNN Final Loss (2.49)')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Fine-tuned CNN Training (8 epochs)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("results/finetuned_8epochs_training.png", dpi=150)
print(f"\nTraining curve saved to: results/finetuned_8epochs_training.png")

# Save updated results
results = [
    {'Model': 'Frozen CNN', 'BLEU-1': frozen_results['BLEU-1'], 'BLEU-4': frozen_results['BLEU-4'], 'Epochs': 8},
    {'Model': 'Fine-tuned CNN (8 epochs)', 'BLEU-1': bleu1_finetuned, 'BLEU-4': bleu4_finetuned, 'Epochs': 8}
]

df_results = pd.DataFrame(results)
df_results.to_csv("results/cnn_tuning_8epochs_comparison.csv", index=False)
print(f"Updated results saved to: results/cnn_tuning_8epochs_comparison.csv")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
if improvement_bleu4 > 0.01:
    print(f"✅ Fine-tuning for 8 epochs improves BLEU-4 by {improvement_bleu4:.4f} ({improvement_bleu4/frozen_results['BLEU-4']*100:.1f}%)")
    print("   Recommendation: Fine-tuning is beneficial for PixStory")
else:
    print(f"⚠️ Fine-tuning for 8 epochs provides minimal BLEU-4 improvement: {improvement_bleu4:.4f}")
    print("   Recommendation: Consider cost-benefit for deployment")

print("\n✅ Fine-tuning re-run completed!")
