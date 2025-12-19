#!/usr/bin/env python3
"""Experiment 2: Hidden Size Comparison (256 vs 512)"""
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

# Load data
df = pd.read_csv(captions_file)
print(f"Total captions: {len(df)}")
unique_images = df['image'].unique()
train_imgs, temp_imgs = train_test_split(unique_images, test_size=0.2, random_state=42)
val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

# Build vocabulary with min_freq=2
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

# Dataset class
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

# ============ MODELS (FIXED ARCHITECTURE) ============
class CNNEncoder(nn.Module):
    def __init__(self, embed_size, fine_tune=False):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        if fine_tune:
            for param in list(self.features.parameters())[-30:]: param.requires_grad = True
        else:
            for param in self.features.parameters(): param.requires_grad = False
        self.projection = nn.Sequential(nn.Linear(resnet.fc.in_features, embed_size), 
                                       nn.BatchNorm1d(embed_size), nn.ReLU(), nn.Dropout(0.3))
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.projection(x)

class DecoderGRU(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, word2idx, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)  # NO CONCATENATION
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.word2idx = word2idx
        self.vocab_size = vocab_size
        self.feature_projection = nn.Linear(embed_size, hidden_size) if embed_size != hidden_size else nn.Identity()

    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        hidden = self.feature_projection(features).unsqueeze(0)
        output, _ = self.gru(embeddings, hidden)  # NO CONCATENATION
        output = self.dropout(output)
        return self.linear(output)

    def predict(self, features, max_len=25, device='cuda'):
        batch_size = features.size(0)
        hidden = self.feature_projection(features).unsqueeze(0)
        inputs = torch.tensor([[self.word2idx['<start>']]] * batch_size, device=device)
        seq = []
        for _ in range(max_len):
            embeddings = self.embed(inputs)
            output, hidden = self.gru(embeddings, hidden)  # NO CONCATENATION
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
    else:
        params = decoder.parameters()
        lr = 1e-3
    optimizer = torch.optim.Adam(params, lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=word2idx['<pad>'])
    
    os.makedirs("models", exist_ok=True)
    
    for epoch in range(epochs):
        encoder.train(); decoder.train()
        train_loss = 0
        for images, captions in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images, captions = images.to(device), captions.to(device)
            features = encoder(images)
            outputs = decoder(features, captions[:, :-1])
            loss = criterion(outputs.reshape(-1, vocab_size), captions[:, 1:].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            optimizer.step()
            train_loss += loss.item()
        
        avg_train = train_loss/len(train_loader)
        print(f"Epoch {epoch+1}: Train Loss: {avg_train:.4f}")
    
    torch.save({
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'word2idx': word2idx,
        'idx2word': idx2word,
        'vocab_size': vocab_size,
        'hidden_size': decoder.gru.hidden_size
    }, f"models/{save_name}.pth")
    print(f"Model saved: models/{save_name}.pth")
    return encoder, decoder

# ============ MULTI-REFERENCE EVALUATION ============
def evaluate_multi_ref(encoder, decoder, test_df, image_dir, transform, word2idx, idx2word, device='cuda'):
    encoder.eval(); decoder.eval()
    
    # Group all references by image
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
                bleu1 = sentence_bleu(refs_list, pred_tokens, weights=(1, 0, 0, 0))
                bleu4 = sentence_bleu(refs_list, pred_tokens, smoothing_function=smoothie)
                bleu1_scores.append(bleu1); bleu4_scores.append(bleu4)
    
    return np.mean(bleu1_scores), np.mean(bleu4_scores)

# ============ RUN EXPERIMENT 2 ============
print("\n" + "="*60)
print("EXPERIMENT 2: HIDDEN SIZE COMPARISON (256 vs 512)")
print("="*60)

os.makedirs("results", exist_ok=True)
results = []
test_df = df[df['image'].isin(test_imgs)].copy()

# Test hidden sizes
hidden_sizes = [256, 512]

for hidden_size in hidden_sizes:
    print(f"\n{'='*50}")
    print(f"Training GRU with hidden_size={hidden_size}")
    print(f"{'='*50}")
    
    model_name = f"exp2_gru_{hidden_size}"
    model_path = f"models/{model_name}.pth"
    
    encoder = CNNEncoder(512, fine_tune=False).to(device)
    decoder = DecoderGRU(512, hidden_size, vocab_size, word2idx).to(device)
    
    if os.path.exists(model_path):
        print(f"Loading existing model {model_name}...")
        checkpoint = torch.load(model_path, map_location=device)
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
    else:
        print(f"Training new model with hidden_size={hidden_size}...")
        encoder, decoder = train_model(
            encoder, decoder, train_loader, val_loader,
            epochs=8, fine_tune=False, save_name=model_name
        )
    
    # Evaluate
    bleu1, bleu4 = evaluate_multi_ref(
        encoder, decoder, test_df, image_dir, 
        transform, word2idx, idx2word, device
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in decoder.parameters())
    
    print(f"\nResults for hidden_size={hidden_size}:")
    print(f"  BLEU-1: {bleu1:.4f}")
    print(f"  BLEU-4: {bleu4:.4f}")
    print(f"  Parameters: {total_params:,}")
    
    results.append({
        'hidden_size': hidden_size,
        'BLEU-1': bleu1,
        'BLEU-4': bleu4,
        'parameters': total_params,
        'model_name': model_name
    })

# Print comparison
print("\n" + "="*60)
print("HIDDEN SIZE COMPARISON RESULTS")
print("="*60)
print(f"{'Hidden Size':<12} {'BLEU-1':<10} {'BLEU-4':<10} {'Parameters':<12}")
print(f"{'-'*50}")
for r in results:
    print(f"{r['hidden_size']:<12} {r['BLEU-1']:<10.4f} {r['BLEU-4']:<10.4f} {r['parameters']:<12,}")

# Save results
df_results = pd.DataFrame(results)
df_results.to_csv("results/hidden_size_comparison.csv", index=False)
print(f"\nResults saved to: results/hidden_size_comparison.csv")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# BLEU scores
sizes = [r['hidden_size'] for r in results]
bleu1_vals = [r['BLEU-1'] for r in results]
bleu4_vals = [r['BLEU-4'] for r in results]

ax1.plot(sizes, bleu1_vals, 'o-', linewidth=2, markersize=8, label='BLEU-1')
ax1.plot(sizes, bleu4_vals, 's-', linewidth=2, markersize=8, label='BLEU-4')
ax1.set_xlabel('Hidden Size')
ax1.set_ylabel('BLEU Score')
ax1.set_title('BLEU vs Hidden Size')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0.40, color='r', linestyle='--', alpha=0.5, label='Baseline BLEU-1')
ax1.axhline(y=0.10, color='g', linestyle='--', alpha=0.5, label='Baseline BLEU-4')

# Parameters
params = [r['parameters'] for r in results]
ax2.bar([str(s) for s in sizes], params, color=['skyblue', 'lightcoral'])
ax2.set_xlabel('Hidden Size')
ax2.set_ylabel('Number of Parameters')
ax2.set_title('Model Size vs Hidden Size')
for i, v in enumerate(params):
    ax2.text(i, v + 1000, f'{v:,}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig("results/hidden_size_comparison.png", dpi=150)
print("Plot saved to: results/hidden_size_comparison.png")

# Analysis
best_model = max(results, key=lambda x: x['BLEU-4'])
print("\n" + "="*60)
print("EXPERIMENT 2 SUMMARY")
print("="*60)
print(f"Best hidden size: {best_model['hidden_size']}")
print(f"Best BLEU-4: {best_model['BLEU-4']:.4f}")
print(f"Improvement (512 vs 256): {(results[1]['BLEU-4'] - results[0]['BLEU-4']):.4f}")

if results[1]['BLEU-4'] - results[0]['BLEU-4'] > 0.01:
    print("Recommendation: Use hidden_size=512 (worth the extra parameters)")
else:
    print("Recommendation: Consider hidden_size=256 (similar performance, fewer parameters)")

print("\n✅ Experiment 2 completed!")
