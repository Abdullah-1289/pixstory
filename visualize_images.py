#!/usr/bin/env python3
"""PixStory Qualitative Image Visualizations ONLY"""
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
import kagglehub
import random

# Add current directory to path
sys.path.insert(0, '.')

def import_model_classes():
    """Import model classes from your experiment files"""
    if os.path.exists('run_rnn_only.py'):
        from run_rnn_only import CNNEncoder, DecoderGRU, DecoderLSTM
        print("✓ Imported from run_rnn_only.py")
        return CNNEncoder, DecoderGRU, DecoderLSTM
    else:
        print("❌ run_rnn_only.py not found!")
        exit(1)

def load_model(model_path, device='cuda'):
    """Load trained model"""
    print(f"Loading: {os.path.basename(model_path)}")
    
    CNNEncoder, DecoderGRU, DecoderLSTM = import_model_classes()
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get parameters
    embed_size = checkpoint.get('embed_size', 512)
    hidden_size = checkpoint.get('hidden_size', 256)
    vocab_size = checkpoint.get('vocab_size', 5000)
    word2idx = checkpoint['word2idx']
    
    # Create encoder
    encoder = CNNEncoder(embed_size, fine_tune=False).to(device)
    
    # Create decoder (GRU or LSTM)
    if 'lstm' in model_path.lower():
        decoder = DecoderLSTM(embed_size, hidden_size, vocab_size, word2idx).to(device)
    else:
        decoder = DecoderGRU(embed_size, hidden_size, vocab_size, word2idx).to(device)
    
    # Load state dicts
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    
    # Get idx2word
    idx2word = checkpoint.get('idx2word', {})
    if not idx2word:
        idx2word = {idx: word for word, idx in word2idx.items()}
    
    encoder.eval()
    decoder.eval()
    
    print(f"  ✓ Model loaded (hidden_size={hidden_size}, vocab={vocab_size})")
    return encoder, decoder, idx2word, word2idx

def generate_caption(encoder, decoder, image_tensor, idx2word, word2idx, device='cuda'):
    """Generate caption for an image"""
    with torch.no_grad():
        features = encoder(image_tensor)
        
        # Try predict method
        if hasattr(decoder, 'predict'):
            pred_ids = decoder.predict(features, device=device)
        else:
            # Manual decoding
            start_token = word2idx.get('<start>', 1)
            inputs = torch.tensor([[start_token]], device=device)
            hidden = None
            pred_ids = []
            
            for _ in range(20):
                output, hidden = decoder(inputs, features, hidden)
                _, predicted = output.max(2)
                word_id = predicted.item()
                
                if word_id == word2idx.get('<end>', 2):
                    break
                
                pred_ids.append(word_id)
                inputs = predicted
        
        # Convert to text
        if isinstance(pred_ids, torch.Tensor):
            pred_ids = pred_ids[0].tolist()
        
        ai_words = []
        for word_id in pred_ids:
            if word_id in idx2word:
                word = idx2word[word_id]
                if word not in ['<start>', '<end>', '<pad>', '<unk>']:
                    ai_words.append(word)
        
        return ' '.join(ai_words).capitalize()

def create_single_visualization(example_num, image, ai_caption, human_captions, img_name, output_dir):
    """Create one clean image+caption visualization"""
    fig, (ax_img, ax_text) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Left: Image
    ax_img.imshow(image)
    ax_img.set_title(f'Image {example_num}', fontsize=14, fontweight='bold', pad=10)
    ax_img.axis('off')
    
    # Right: Captions
    text_content = f"🤖 AI-GENERATED CAPTION:\n\n{ai_caption}\n\n"
    text_content += "─" * 40 + "\n\n"
    text_content += "👥 HUMAN REFERENCE CAPTIONS:\n\n"
    
    for j, cap in enumerate(human_captions):
        text_content += f"{j+1}. {cap}\n\n"
    
    ax_text.text(0.05, 0.5, text_content, fontsize=12,
                ha='left', va='center', wrap=True,
                transform=ax_text.transAxes)
    ax_text.axis('off')
    
    # Footer
    plt.figtext(0.5, 0.02, 
               f"Model: GRU-256 | BLEU-1: 0.573 | Image: {img_name[:20]}...",
               ha='center', fontsize=10, style='italic')
    
    plt.suptitle(f'PixStory Image Captioning - Example {example_num}/10', 
                fontsize=16, fontweight='bold', y=0.97)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save
    plt.savefig(f'{output_dir}/example_{example_num:02d}.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """Main function to generate qualitative examples"""
    print("="*60)
    print("PixStory Qualitative Image Visualizations")
    print("="*60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'results/qualitative_images'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model_path = 'models/exp2_gru_256.pth'
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    encoder, decoder, idx2word, word2idx = load_model(model_path, device)
    
    # Download data
    print("\nDownloading dataset...")
    try:
        path = kagglehub.dataset_download("adityajn105/flickr8k")
        image_dir = f"{path}/Images"
        captions_file = f"{path}/captions.txt"
        
        df = pd.read_csv(captions_file)
        print(f"✓ Loaded {len(df)} captions")
        
        # Get test images (last 10%)
        unique_images = df['image'].unique()
        test_images = list(unique_images[-len(unique_images)//10:])
        print(f"✓ Test images: {len(test_images)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # Image preprocessing
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Select 10 random images (change seed for different images)
    print(f"\nSelecting 10 random images...")
    random.seed(100)  # Change this number for different images
    selected_images = random.sample(test_images, min(10, len(test_images)))
    
    print(f"Generating captions...")
    
    successful = 0
    for i, img_name in enumerate(selected_images):
        try:
            # Find image
            img_path = None
            base_name = img_name.split('.')[0]
            
            for ext in ['.jpg', '.jpeg', '.png']:
                test_path = os.path.join(image_dir, base_name + ext)
                if os.path.exists(test_path):
                    img_path = test_path
                    break
            
            if not img_path:
                print(f"  {i+1}. ✗ Image not found: {img_name}")
                continue
            
            # Load image
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            # Generate caption
            ai_caption = generate_caption(encoder, decoder, image_tensor, idx2word, word2idx, device)
            
            # Get human captions
            human_captions = df[df['image'] == img_name]['caption'].tolist()[:2]
            
            # Create visualization
            create_single_visualization(i+1, image, ai_caption, human_captions, 
                                       os.path.basename(img_path), output_dir)
            
            successful += 1
            print(f"  {i+1}. ✓ Created: {os.path.basename(img_path)}")
            
        except Exception as e:
            print(f"  {i+1}. ✗ Error: {str(e)[:50]}")
    
    # Summary
    print(f"\n" + "="*60)
    print(f"RESULTS: Created {successful}/10 qualitative examples")
    print(f"Location: {output_dir}/")
    
    if successful >= 10:
        print("✅ SUCCESS: All requirements met!")
    
    print("="*60)

if __name__ == "__main__":
    main()