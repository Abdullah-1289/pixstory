#!/usr/bin/env python3
"""Run only RNN comparison (GRU vs LSTM)"""
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
print("Downloading dataset...")
path = kagglehub.dataset_download("adityajn105/flickr8k")
image_dir = f"{path}/Images"
captions_file = f"{path}/captions.txt"
print(f"Dataset at: {path}")

# Load data
print("Loading data...")
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
vocab += [word for word, count in word_counts.items() if count >= 2]  # CHANGED: was 5

word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for word, idx in word2idx.items()}
vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")

# Dataset class and loaders
class FlickrDataset(Dataset):
    def __init__(self, df, word2idx, image_dir, max_len=25, transform=None):
        self.df = df; self.word2idx = word2idx; self.image_dir = image_dir; self.max_len = max_len; self.transform = transform
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

transform = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
train_dataset = FlickrDataset(df[df['image'].isin(train_imgs)], word2idx, image_dir, transform=transform)
val_dataset = FlickrDataset(df[df['image'].isin(val_imgs)], word2idx, image_dir, transform=transform)
test_dataset = FlickrDataset(df[df['image'].isin(test_imgs)], word2idx, image_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
print(f"Data loaded: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")

# ============ MODELS (WITH FIXED ARCHITECTURE) ============
class CNNEncoder(nn.Module):
    def __init__(self, embed_size, fine_tune=False):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        if fine_tune:
            for param in list(self.features.parameters())[-30:]: param.requires_grad = True
        else:
            for param in self.features.parameters(): param.requires_grad = False
        self.projection = nn.Sequential(nn.Linear(resnet.fc.in_features, embed_size), nn.BatchNorm1d(embed_size), nn.ReLU(), nn.Dropout(0.3))
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.projection(x)

class DecoderGRU(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, word2idx, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)  # FIXED: NO concatenation
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.word2idx = word2idx
        self.vocab_size = vocab_size
        self.feature_projection = nn.Linear(embed_size, hidden_size) if embed_size != hidden_size else nn.Identity()

    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        hidden = self.feature_projection(features).unsqueeze(0)
        output, _ = self.gru(embeddings, hidden)  # NO concatenation
        output = self.dropout(output)
        return self.linear(output)

    def predict(self, features, max_len=25, device='cuda'):
        batch_size = features.size(0)
        hidden = self.feature_projection(features).unsqueeze(0)
        inputs = torch.tensor([[self.word2idx['<start>']]] * batch_size, device=device)
        seq = []
        
        for _ in range(max_len):
            embeddings = self.embed(inputs)
            output, hidden = self.gru(embeddings, hidden)  # NO concatenation
            predicted = self.linear(output.squeeze(1)).argmax(1)
            seq.append(predicted.unsqueeze(1))
            inputs = predicted.unsqueeze(1)
            if (predicted == self.word2idx['<end>']).all():
                break
        
        return torch.cat(seq, 1)

class DecoderLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, word2idx, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)  # FIXED: NO concatenation
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.word2idx = word2idx

    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        hidden = (features.unsqueeze(0), torch.zeros_like(features).unsqueeze(0))
        output, _ = self.lstm(embeddings, hidden)  # NO concatenation
        output = self.dropout(output)
        return self.linear(output)

    def predict(self, features, max_len=25, device='cuda'):
        batch_size = features.size(0)
        hidden = (features.unsqueeze(0), torch.zeros_like(features).unsqueeze(0))
        inputs = torch.tensor([[self.word2idx['<start>']]] * batch_size, device=device)
        seq = []
        for _ in range(max_len):
            embeddings = self.embed(inputs)
            output, hidden = self.lstm(embeddings, hidden)  # NO concatenation
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
        'vocab_size': vocab_size
    }, f"models/{save_name}.pth")
    print(f"Model saved: models/{save_name}.pth")
    return encoder, decoder

# ============ MULTI-REFERENCE EVALUATION ============
def evaluate_multi_ref(encoder, decoder, test_df, image_dir, transform, word2idx, idx2word, device='cuda'):
    """Evaluate with ALL 5 reference captions per image"""
    encoder.eval()
    decoder.eval()
    
    # Group all references by image
    image_to_refs = {}
    for _, row in test_df.iterrows():
        img_name = row['image']
        tokens = nltk.word_tokenize(row['caption'].lower())
        image_to_refs.setdefault(img_name, []).append(tokens)
    
    bleu1_scores = []
    bleu4_scores = []
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
                bleu1_scores.append(bleu1)
                bleu4_scores.append(bleu4)
    
    return np.mean(bleu1_scores), np.mean(bleu4_scores)

# ============ RUN RNN COMPARISON ONLY ============
print("\n" + "="*60)
print("EXPERIMENT 1: RNN TYPE COMPARISON (LSTM vs GRU)")
print("="*60)

os.makedirs("results", exist_ok=True)
results = []

# Prepare test DataFrame for multi-reference evaluation
test_df = df[df['image'].isin(test_imgs)].copy()

print("\n1. Training GRU model...")
encoder_gru = CNNEncoder(512, fine_tune=False).to(device)
decoder_gru = DecoderGRU(512, 512, vocab_size, word2idx).to(device)

# Check if model exists
if os.path.exists("models/exp1_gru_fixed.pth"):
    print("   Loading existing GRU model...")
    checkpoint = torch.load("models/exp1_gru_fixed.pth", map_location=device)
    encoder_gru.load_state_dict(checkpoint['encoder'])
    decoder_gru.load_state_dict(checkpoint['decoder'])
else:
    print("   Training new GRU model with fixed architecture...")
    encoder_gru, decoder_gru = train_model(
        encoder_gru, decoder_gru, train_loader, val_loader,
        epochs=8, fine_tune=False, save_name="exp1_gru_fixed"
    )

# Evaluate GRU with multi-reference BLEU
bleu1_gru, bleu4_gru = evaluate_multi_ref(
    encoder_gru, decoder_gru, test_df, image_dir, 
    transform, word2idx, idx2word, device
)
print(f"   GRU Results: BLEU-1={bleu1_gru:.4f}, BLEU-4={bleu4_gru:.4f}")
results.append(('GRU', bleu1_gru, bleu4_gru))

print("\n2. Training LSTM model...")
encoder_lstm = CNNEncoder(512, fine_tune=False).to(device)
decoder_lstm = DecoderLSTM(512, 512, vocab_size, word2idx).to(device)

# Check if model exists
if os.path.exists("models/exp1_lstm_fixed.pth"):
    print("   Loading existing LSTM model...")
    checkpoint = torch.load("models/exp1_lstm_fixed.pth", map_location=device)
    encoder_lstm.load_state_dict(checkpoint['encoder'])
    decoder_lstm.load_state_dict(checkpoint['decoder'])
else:
    print("   Training new LSTM model with fixed architecture...")
    encoder_lstm, decoder_lstm = train_model(
        encoder_lstm, decoder_lstm, train_loader, val_loader,
        epochs=8, fine_tune=False, save_name="exp1_lstm_fixed"
    )

# Evaluate LSTM with multi-reference BLEU
bleu1_lstm, bleu4_lstm = evaluate_multi_ref(
    encoder_lstm, decoder_lstm, test_df, image_dir,
    transform, word2idx, idx2word, device
)
print(f"   LSTM Results: BLEU-1={bleu1_lstm:.4f}, BLEU-4={bleu4_lstm:.4f}")
results.append(('LSTM', bleu1_lstm, bleu4_lstm))

# Print comparison
print("\n" + "="*60)
print("RNN TYPE COMPARISON RESULTS")
print("="*60)
print(f"{'Model':<10} {'BLEU-1':<10} {'BLEU-4':<10}")
print(f"{'-'*30}")
for model, b1, b4 in results:
    print(f"{model:<10} {b1:<10.4f} {b4:<10.4f}")

# Save results
import pandas as pd
df_results = pd.DataFrame(results, columns=['Model', 'BLEU-1', 'BLEU-4'])
df_results.to_csv("results/rnn_comparison.csv", index=False)
print(f"\nResults saved to: results/rnn_comparison.csv")

# Simple plot
plt.figure(figsize=(8, 5))
x = np.arange(len(results))
width = 0.35

plt.bar(x - width/2, [r[1] for r in results], width, label='BLEU-1')
plt.bar(x + width/2, [r[2] for r in results], width, label='BLEU-4')
plt.xlabel('Model')
plt.ylabel('BLEU Score')
plt.title('RNN Type Comparison')
plt.xticks(x, [r[0] for r in results])
plt.axhline(y=0.40, color='r', linestyle='--', alpha=0.5, label='Baseline BLEU-1')
plt.axhline(y=0.10, color='g', linestyle='--', alpha=0.5, label='Baseline BLEU-4')
plt.legend()
plt.tight_layout()
plt.savefig("results/rnn_comparison.png", dpi=150)
print("Plot saved to: results/rnn_comparison.png")

print("\n✅ Experiment 1 completed!")
