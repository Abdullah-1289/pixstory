import torch
import matplotlib.pyplot as plt
from PIL import Image
import os
import json

def visualize_predictions(encoder, decoder, test_images, image_dir, transform,
                         word2idx, idx2word, save_dir="results/qualitative",
                         beam_width=3, max_examples=10):
    """
    Generate visualizations of model predictions vs ground truth.
    """
    device = next(encoder.parameters()).device
    encoder.eval()
    decoder.eval()
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Load test captions for references
    import pandas as pd
    from original_implementation import df, test_imgs  # Import from your main file
    
    test_df = df[df['image'].isin(test_images[:max_examples])]
    
    # Group references
    image_to_refs = {}
    for _, row in test_df.iterrows():
        img_name = row['image']
        caption = row['caption']
        image_to_refs.setdefault(img_name, []).append(caption)
    
    results = []
    
    for i, img_name in enumerate(test_images[:max_examples]):
        if img_name not in image_to_refs:
            continue
            
        # Load image
        img_path = f"{image_dir}/{img_name}"
        image = Image.open(img_path).convert('RGB')
        
        # Generate caption
        image_t = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            features = encoder(image_t)
            
            if beam_width > 1 and hasattr(decoder, 'beam_search_predict'):
                pred_ids = decoder.beam_search_predict(features, beam_width=beam_width)
            else:
                pred_ids = decoder.predict(features)
            
            # Convert to words
            pred_words = []
            for idx in pred_ids[0]:
                word = idx2word[idx.item()]
                if word in ['<start>', '<end>', '<pad>', '<unk>']:
                    continue
                pred_words.append(word)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left: Image with predicted caption
        axes[0].imshow(image)
        axes[0].axis('off')
        pred_text = ' '.join(pred_words)
        axes[0].set_title(f'Predicted:\\n{pred_text}', fontsize=11, pad=10)
        
        # Right: Reference captions
        refs = image_to_refs[img_name][:3]  # First 3 references
        ref_text = '\\n\\n'.join([f'Ref {j+1}: {ref}' for j, ref in enumerate(refs)])
        axes[1].text(0, 0.5, ref_text, fontsize=10, verticalalignment='center')
        axes[1].axis('off')
        axes[1].set_title('Human References', fontsize=11, pad=10)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/example_{i+1:02d}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Store for summary
        results.append({
            'image': img_name,
            'predicted': pred_text,
            'references': refs
        })
        
        print(f"Generated visualization {i+1}/{min(max_examples, len(test_images))}")
    
    # Create HTML summary
    html_content = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>PixStory Captioning Results</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .example { margin: 30px 0; border: 1px solid #ddd; padding: 20px; }
            .image-row { display: flex; }
            .image-container { flex: 1; margin: 10px; }
            .caption { margin-top: 10px; font-style: italic; }
            .refs { margin-top: 15px; color: #555; }
        </style>
    </head>
    <body>
        <h1>PixStory Image Captioning - Qualitative Results</h1>
    '''
    
    for i, result in enumerate(results):
        html_content += f'''
        <div class="example">
            <h2>Example {i+1}: {result['image']}</h2>
            <div class="image-row">
                <div class="image-container">
                    <img src="example_{i+1:02d}.png" width="100%">
                </div>
            </div>
            <div class="caption">
                <strong>Predicted:</strong> {result['predicted']}
            </div>
            <div class="refs">
                <strong>References:</strong><br>
                {''.join([f'{j+1}. {ref}<br>' for j, ref in enumerate(result['references'])])}
            </div>
        </div>
        '''
    
    html_content += '''
    </body>
    </html>
    '''
    
    with open(f"{save_dir}/summary.html", 'w') as f:
        f.write(html_content)
    
    print(f"\n✅ Generated {len(results)} visualizations in {save_dir}/")
    print(f"✅ HTML summary: {save_dir}/summary.html")
    
    return results

if __name__ == "__main__":
    # This will be run separately
    print("Run this script after training your model.")
    print("First, modify it to load your specific model and data.")

