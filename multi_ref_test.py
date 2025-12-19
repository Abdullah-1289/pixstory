# -*- coding: utf-8 -*-
"""Multi-Reference BLEU Test for PixStory GRU Model"""
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
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
import nltk
nltk.download('punkt', quiet=True)
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import warnings
warnings.filterwarnings('ignore')

# ============ SETUP ============
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Download dataset (if not already downloaded)
import kagglehub
print("Loading dataset...")
path = kagglehub.dataset_download("adityajn105/flickr8k")
image_dir = f"{path}/Images"
captions_file = f"{path}/captions.txt"

# Load data
df = pd.read_csv(captions_file)
print(f"Total captions: {len(df)}")

# Load the SAME splits you used originally
# You must use the exact same split to match your trained model
print("Recreating original 80/10/10 splits...")
unique_images = df['image'].unique()

# IMPORTANT: Use the SAME random_state (42) as in original_implementation.py
from sklearn.model_selection import train_test_split
train_imgs, temp_imgs = train_test_split(unique_images, test_size=0.2, random_state=42)
val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

print(f"Test images: {len(test_imgs)}")

# Build vocabulary (must match original)
print("Building vocabulary (must match training)...")
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

# ============ MODEL DEFINITIONS (MUST MATCH ORIGINAL) ============
# Copy the EXACT same model classes from original_implementation.py
class CNNEncoder(nn.Module):
    def __init__(self, embed_size, fine_tune=False):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        if fine_tune:
            for param in list(self.features.parameters())[-30:]: 
                param.requires_grad = True
        else:
            for param in self.features.parameters(): 
                param.requires_grad = False
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
        self.gru = nn.GRU(embed_size + embed_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.word2idx = word2idx
        self.vocab_size = vocab_size
        self.feature_projection = nn.Linear(embed_size, hidden_size) if embed_size != hidden_size else nn.Identity()

    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        hidden_features = self.feature_projection(features)
        features_expanded = features.unsqueeze(1)
        features_expanded = features_expanded.repeat(1, embeddings.size(1), 1)
        gru_input = torch.cat([embeddings, features_expanded], dim=2)
        hidden = hidden_features.unsqueeze(0)
        output, _ = self.gru(gru_input, hidden)
        output = self.dropout(output)
        return self.linear(output)

    def predict(self, features, max_len=25, device='cuda'):
        batch_size = features.size(0)
        hidden_features = self.feature_projection(features)
        hidden = hidden_features.unsqueeze(0)
        inputs = torch.tensor([[self.word2idx['<start>']]] * batch_size, device=device)
        seq = []
        
        for _ in range(max_len):
            embeddings = self.embed(inputs)
            features_expanded = features.unsqueeze(1)
            gru_input = torch.cat([embeddings, features_expanded], dim=2)
            output, hidden = self.gru(gru_input, hidden)
            predicted = self.linear(output.squeeze(1)).argmax(1)
            seq.append(predicted.unsqueeze(1))
            inputs = predicted.unsqueeze(1)
            
            if (predicted == self.word2idx['<end>']).all():
                break
        
        return torch.cat(seq, 1)

# ============ MULTI-REFERENCE EVALUATION ============
def evaluate_multi_ref():
    """
    Evaluates the saved GRU model using ALL 5 reference captions per image.
    """
    print("\n" + "="*60)
    print("MULTI-REFERENCE BLEU EVALUATION")
    print("="*60)
    
    # 1. Load the trained model
    model_path = "models/exp1_gru.pth"
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Please train the model first using original_implementation.py")
        return
    
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Recreate model architecture
    encoder = CNNEncoder(512, fine_tune=False).to(device)
    decoder = DecoderGRU(512, 512, vocab_size, word2idx, dropout=0.3).to(device)
    
    # Load weights
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    
    encoder.eval()
    decoder.eval()
    print("Model loaded successfully.")
    
    # 2. Prepare test data with ALL captions grouped by image
    test_df = df[df['image'].isin(test_imgs)].copy()
    
    # Tokenization function (must match original preprocessing)
    def caption_to_tokens(caption):
        tokens = nltk.word_tokenize(caption.lower())
        # Don't add <start>, <end>, <pad> for BLEU evaluation
        return tokens
    
    # Group all reference captions by image
    image_to_refs = {}
    for _, row in test_df.iterrows():
        img_name = row['image']
        tokens = caption_to_tokens(row['caption'])
        if img_name not in image_to_refs:
            image_to_refs[img_name] = []
        image_to_refs[img_name].append(tokens)
    
    print(f"Test images: {len(image_to_refs)}")
    print(f"Reference captions per image: {len(list(image_to_refs.values())[0])}")
    
    # 3. Image transform (must match original)
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 4. Evaluate each test image
    bleu1_scores = []
    bleu4_scores = []
    smoothie = SmoothingFunction().method1
    captions_generated = []  # Store for display
    
    print("\nGenerating captions and computing BLEU...")
    with torch.no_grad():
        for img_name, refs_list in tqdm(image_to_refs.items(), desc="Processing images"):
            # Load and preprocess image
            img_path = f"{image_dir}/{img_name}"
            try:
                image = Image.open(img_path).convert('RGB')
            except:
                print(f"Warning: Could not load {img_path}, skipping")
                continue
            
            image_t = transform(image).unsqueeze(0).to(device)
            
            # Generate caption
            features = encoder(image_t)
            pred_ids = decoder.predict(features, device=device, max_len=25)
            
            # Convert prediction to words
            pred_tokens = []
            for idx in pred_ids[0]:
                word = idx2word[idx.item()]
                if word in ['<start>', '<end>', '<pad>']:
                    continue
                if word == '<unk>':
                    break
                pred_tokens.append(word)
            
            # Skip if prediction is empty
            if not pred_tokens:
                continue
            
            # Compute BLEU with ALL references
            # Note: We use smoothing for BLEU-4 to handle short sentences
            bleu1 = sentence_bleu(refs_list, pred_tokens, weights=(1, 0, 0, 0))
            bleu4 = sentence_bleu(refs_list, pred_tokens, smoothing_function=smoothie)
            
            bleu1_scores.append(bleu1)
            bleu4_scores.append(bleu4)
            
            # Store first few examples for display
            if len(captions_generated) < 5:
                captions_generated.append({
                    'image': img_name,
                    'predicted': ' '.join(pred_tokens),
                    'references': [' '.join(ref) for ref in refs_list[:2]],  # Show first 2 refs
                    'bleu1': bleu1,
                    'bleu4': bleu4
                })
    
    # 5. Display results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    print(f"\nNumber of images evaluated: {len(bleu1_scores)}")
    print(f"Mean BLEU-1: {np.mean(bleu1_scores):.4f}")
    print(f"Mean BLEU-4: {np.mean(bleu4_scores):.4f}")
    print(f"BLEU-1 Std: {np.std(bleu1_scores):.4f}")
    print(f"BLEU-4 Std: {np.std(bleu4_scores):.4f}")
    
    # Compare with your old single-reference scores
    print("\n" + "-"*40)
    print("COMPARISON WITH SINGLE-REFERENCE SCORING")
    print("(Your original implementation used only 1 reference)")
    print("-"*40)
    print("Expected improvement: +0.05 to +0.15 in BLEU-1")
    
    # Show example captions
    print("\n" + "="*60)
    print("EXAMPLE CAPTIONS (first 5 images)")
    print("="*60)
    
    for i, example in enumerate(captions_generated):
        print(f"\nExample {i+1}: {example['image']}")
        print(f"  Predicted: {example['predicted']}")
        print(f"  Reference 1: {example['references'][0]}")
        print(f"  Reference 2: {example['references'][1] if len(example['references']) > 1 else 'N/A'}")
        print(f"  BLEU-1: {example['bleu1']:.4f}, BLEU-4: {example['bleu4']:.4f}")
    
    # Save results to file
    results = {
        'bleu1_mean': float(np.mean(bleu1_scores)),
        'bleu4_mean': float(np.mean(bleu4_scores)),
        'bleu1_std': float(np.std(bleu1_scores)),
        'bleu4_std': float(np.std(bleu4_scores)),
        'n_images': len(bleu1_scores),
        'examples': captions_generated
    }
    
    with open('multi_ref_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: multi_ref_results.json")

# ============ RUN ============
if __name__ == "__main__":
    evaluate_multi_ref()
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("1. If BLEU-1 > 0.35, your model is better than you thought!")
    print("2. Next, fix the decoder architecture (remove feature concatenation)")
    print("3. Then add attention mechanism for further improvement")
    print("="*60)