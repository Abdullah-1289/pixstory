# src/utils/data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import pandas as pd
import json
from sklearn.model_selection import train_test_split
import nltk
from collections import Counter
import kagglehub
from config import SEED, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS

nltk.download('punkt_tab', quiet=True)

class FlickrDataset(Dataset):
    def __init__(self, df, word2idx, image_dir, max_len=25, transform=None):
        self.df = df.reset_index(drop=True)
        self.word2idx = word2idx
        self.image_dir = image_dir
        self.max_len = max_len
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = f"{self.image_dir}/{row['image']}"
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color='white')
        
        if self.transform:
            image = self.transform(image)
        
        # Process caption
        caption = str(row['caption']).lower()
        tokens = nltk.word_tokenize(caption)
        tokens = ['<start>'] + tokens + ['<end>']
        
        # Pad or truncate
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        else:
            tokens = tokens + ['<pad>'] * (self.max_len - len(tokens))
        
        # Convert to IDs
        caption_ids = [self.word2idx.get(t, self.word2idx['<unk>']) for t in tokens]
        
        return image, torch.tensor(caption_ids, dtype=torch.long)

def build_vocabulary(train_df, min_freq=5):
    """Build vocabulary from training captions"""
    all_tokens = []
    for caption in train_df['caption']:
        tokens = nltk.word_tokenize(str(caption).lower())
        all_tokens.extend(tokens)
    
    word_counts = Counter(all_tokens)
    
    # Start with special tokens
    vocab = ['<pad>', '<start>', '<end>', '<unk>']
    vocab += [word for word, count in word_counts.items() if count >= min_freq]
    
    # Create mappings
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    vocab_size = len(vocab)
    
    print(f"Vocabulary size: {vocab_size}")
    
    return word2idx, idx2word, vocab_size

def download_and_load_data():
    """Download dataset and load captions"""
    print("Downloading Flickr8k dataset...")
    path = kagglehub.dataset_download("adityajn105/flickr8k")
    image_dir = f"{path}/Images"
    captions_file = f"{path}/captions.txt"
    print(f"Dataset downloaded to: {path}")
    
    df = pd.read_csv(captions_file)
    print(f"Total captions: {len(df)}")
    print(f"Unique images: {df['image'].nunique()}")
    
    return df, image_dir

def get_data_splits(df, test_size=0.2, val_size=0.1):
    """Split data into train/val/test at image level"""
    unique_images = df['image'].unique()
    
    # First split: train vs temp
    train_imgs, temp_imgs = train_test_split(
        unique_images, 
        test_size=test_size, 
        random_state=SEED
    )
    
    # Second split: val vs test (val_size of remaining)
    val_ratio = val_size / (val_size + (test_size - val_size))
    val_imgs, test_imgs = train_test_split(
        temp_imgs, 
        test_size=1-val_ratio, 
        random_state=SEED
    )
    
    # Create DataFrames
    train_df = df[df['image'].isin(train_imgs)]
    val_df = df[df['image'].isin(val_imgs)]
    test_df = df[df['image'].isin(test_imgs)]
    
    print(f"\nData Splits:")
    print(f"  Train: {len(train_imgs)} images, {len(train_df)} captions")
    print(f"  Val:   {len(val_imgs)} images, {len(val_df)} captions")
    print(f"  Test:  {len(test_imgs)} images, {len(test_df)} captions")
    
    # Save splits
    splits = {
        'train': train_imgs.tolist(),
        'val': val_imgs.tolist(),
        'test': test_imgs.tolist()
    }
    return train_df, val_df, test_df, splits

def get_data_loaders():
    """Get all data loaders with proper splits"""
    # Download and load data
    df, image_dir = download_and_load_data()
    
    # Get splits
    train_df, val_df, test_df, splits = get_data_splits(df)
    
    # Build vocabulary
    word2idx, idx2word, vocab_size = build_vocabulary(train_df)
    
    # Define transforms
    transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], 
                   std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = FlickrDataset(train_df, word2idx, image_dir, transform=transform)
    val_dataset = FlickrDataset(val_df, word2idx, image_dir, transform=transform)
    test_dataset = FlickrDataset(test_df, word2idx, image_dir, transform=transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    print(f"\nData loaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'word2idx': word2idx,
        'idx2word': idx2word,
        'vocab_size': vocab_size,
        'splits': splits,
        'image_dir': image_dir
    }