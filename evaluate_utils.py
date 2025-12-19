import torch
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import json
import os

def evaluate_multi_reference(encoder, decoder, test_df, image_dir, transform, 
                            word2idx, idx2word, device='cuda', beam_width=1):
    """
    Evaluate model using ALL reference captions for each image.
    """
    encoder.eval()
    decoder.eval()
    
    # Group all references by image
    image_to_refs = {}
    for _, row in test_df.iterrows():
        img_name = row['image']
        tokens = [idx2word[idx] for idx in row['caption_ids'] 
                 if idx2word[idx] not in ['<start>', '<end>', '<pad>']]
        image_to_refs.setdefault(img_name, []).append(tokens)
    
    bleu1_scores = []
    bleu4_scores = []
    smoothie = SmoothingFunction().method1
    
    results = []
    
    print(f"Evaluating {len(image_to_refs)} images with beam_width={beam_width}...")
    
    with torch.no_grad():
        for img_name, refs_list in tqdm(image_to_refs.items(), desc="Processing"):
            # Load image
            from PIL import Image
            img_path = f"{image_dir}/{img_name}"
            try:
                image = Image.open(img_path).convert('RGB')
            except:
                continue
            
            # Preprocess
            image_t = transform(image).unsqueeze(0).to(device)
            
            # Generate caption
            features = encoder(image_t)
            
            if beam_width > 1 and hasattr(decoder, 'beam_search_predict'):
                pred_ids = decoder.beam_search_predict(features, beam_width=beam_width)
            else:
                pred_ids = decoder.predict(features)
            
            # Convert to tokens
            pred_tokens = []
            for idx in pred_ids[0]:
                word = idx2word[idx.item()]
                if word in ['<start>', '<end>', '<pad>', '<unk>']:
                    continue
                pred_tokens.append(word)
            
            if not pred_tokens:
                continue
            
            # Calculate BLEU with all references
            bleu1 = sentence_bleu(refs_list, pred_tokens, weights=(1, 0, 0, 0))
            bleu4 = sentence_bleu(refs_list, pred_tokens, smoothing_function=smoothie)
            
            bleu1_scores.append(bleu1)
            bleu4_scores.append(bleu4)
            
            # Store example
            if len(results) < 10:  # Keep first 10 for qualitative analysis
                results.append({
                    'image': img_name,
                    'predicted': ' '.join(pred_tokens),
                    'references': [' '.join(ref) for ref in refs_list[:2]],  # First 2 refs
                    'bleu1': float(bleu1),
                    'bleu4': float(bleu4)
                })
    
    # Save detailed results
    output = {
        'bleu1_mean': float(np.mean(bleu1_scores)),
        'bleu4_mean': float(np.mean(bleu4_scores)),
        'bleu1_std': float(np.std(bleu1_scores)),
        'bleu4_std': float(np.std(bleu4_scores)),
        'n_images': len(bleu1_scores),
        'beam_width': beam_width,
        'examples': results
    }
    
    os.makedirs("results", exist_ok=True)
    with open(f"results/evaluation_beam{beam_width}.json", 'w') as f:
        json.dump(output, f, indent=2)
    
    return output

