"""
Data loading utilities for Flickr8k dataset
"""
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import pandas as pd
import nltk
from collections import Counter
import json

def build_vocab(df, min_freq=5):
    """Build vocabulary from training captions"""
    all_tokens = []
    for caption in df['caption']:
        tokens = nltk.word_tokenize(str(caption).lower())
        all_tokens.extend(tokens)
    
    word_counts = Counter(all_tokens)
    vocab = ['<pad>', '<start>', '<end>', '<unk>']
    vocab += [word for word, count in word_counts.items() if count >= min_freq]
    
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word, len(vocab)

class FlickrDataset(Dataset):
    def __init__(self, df, word2idx, image_dir, max_len=25, transform=None):
        self.df = df
        self.word2idx = word2idx
        self.image_dir = image_dir
        self.max_len = max_len
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(f"{self.image_dir}/{row['image']}").convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        caption = row['caption']
        tokens = nltk.word_tokenize(str(caption).lower())
        tokens = ['<start>'] + tokens + ['<end>']
        tokens = tokens[:self.max_len]
        tokens += ['<pad>'] * (self.max_len - len(tokens))
        
        caption_ids = [self.word2idx.get(token, self.word2idx['<unk>']) for token in tokens]
        return image, torch.tensor(caption_ids)

def get_transforms():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
