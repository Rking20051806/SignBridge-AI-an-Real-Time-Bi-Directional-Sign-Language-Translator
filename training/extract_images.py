"""
Extract images from captured JSON dataset and save to folders

Usage:
    python extract_images.py --dataset path/to/asl_captured_dataset.json --output ./asl_dataset

This will create folders like:
    ./asl_dataset/A/A_123456.png
    ./asl_dataset/B/B_789012.png
    ...
"""

import os
import json
import base64
import argparse
from pathlib import Path
from io import BytesIO
from PIL import Image


def extract_images(json_path, output_dir):
    """Extract all images from JSON dataset to folders"""
    
    print(f"Loading dataset from: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    images = data.get('images', [])
    print(f"Found {len(images)} images")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Stats
    stats = {}
    
    for i, img_data in enumerate(images):
        letter = img_data['letter'].upper()
        
        # Create letter folder
        letter_dir = output_path / letter
        letter_dir.mkdir(exist_ok=True)
        
        # Extract image
        try:
            base64_str = img_data['imageData']
            if ',' in base64_str:
                base64_str = base64_str.split(',')[1]
            
            img_bytes = base64.b64decode(base64_str)
            img = Image.open(BytesIO(img_bytes))
            
            # Generate filename
            timestamp = img_data.get('filename', f"{letter}_{i}")
            if not timestamp.endswith('.png'):
                timestamp = f"{letter}_{i}.png"
            
            # Save image
            img_path = letter_dir / timestamp
            img.save(img_path)
            
            # Update stats
            if letter not in stats:
                stats[letter] = 0
            stats[letter] += 1
            
        except Exception as e:
            print(f"  Error extracting image {i} ({letter}): {e}")
            continue
        
        if (i + 1) % 100 == 0:
            print(f"  Extracted {i + 1}/{len(images)} images...")
    
    print("\n✅ Extraction complete!")
    print("\nImages per letter:")
    for letter in sorted(stats.keys()):
        print(f"  {letter}: {stats[letter]}")
    
    print(f"\nTotal: {sum(stats.values())} images")
    print(f"Output directory: {output_path.absolute()}")


def main():
    parser = argparse.ArgumentParser(description='Extract images from captured JSON dataset')
    parser.add_argument('--dataset', type=str, required=True, help='Path to JSON dataset')
    parser.add_argument('--output', type=str, default='./asl_dataset', help='Output directory')
    
    args = parser.parse_args()
    extract_images(args.dataset, args.output)


if __name__ == "__main__":
    main()
