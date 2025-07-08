#!/usr/bin/env python3
"""
Dataset conversion script for RF-DETR fine-tuning.
Converts COCO format dataset to the required structure:
dataset/
├── train/
│   ├── _annotations.coco.json
│   └── images...
├── valid/
│   ├── _annotations.coco.json
│   └── images...
└── test/
    ├── _annotations.coco.json
    └── images...
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from collections import defaultdict
import random

def load_coco_annotation(annotation_path: str) -> Dict:
    """Load COCO annotation file."""
    print(f"Loading annotation file: {annotation_path}")
    with open(annotation_path, 'r') as f:
        coco_data = json.load(f)
    
    print(f"Found {len(coco_data['images'])} images")
    print(f"Found {len(coco_data['annotations'])} annotations")
    print(f"Found {len(coco_data['categories'])} categories")
    
    return coco_data

def create_category_mapping(categories: List[Dict]) -> Dict:
    """Create mapping from category names to IDs."""
    category_map = {}
    for cat in categories:
        category_map[cat['name']] = cat['id']
    
    print("Category mapping:")
    for name, id in category_map.items():
        print(f"  {name}: {id}")
    
    return category_map

def find_image_files(data_dir: str, image_filenames: List[str]) -> Dict[str, str]:
    """Find actual image file paths from the dataset directory."""
    print(f"Searching for images in: {data_dir}")
    
    image_paths = {}
    data_path = Path(data_dir)
    
    # Search in all subdirectories for images
    for image_filename in image_filenames:
        found = False
        
        # First try the exact path relative to data_dir
        exact_path = data_path / image_filename
        if exact_path.is_file():
            image_paths[image_filename] = str(exact_path)
            found = True
        else:
            # If not found, try relative to images subdirectory
            images_path = data_path / "images" / image_filename
            if images_path.is_file():
                image_paths[image_filename] = str(images_path)
                found = True
            else:
                # If still not found, search recursively
                for image_file in data_path.rglob(Path(image_filename).name):
                    if image_file.is_file():
                        image_paths[image_filename] = str(image_file)
                        found = True
                        break
        
        if not found:
            print(f"Warning: Image {image_filename} not found!")
    
    print(f"Found {len(image_paths)} out of {len(image_filenames)} images")
    return image_paths

def split_dataset(images: List[Dict], annotations: List[Dict], 
                 train_ratio: float = 0.7, val_ratio: float = 0.2, test_ratio: float = 0.1) -> Dict:
    """Split dataset into train, validation, and test sets."""
    
    # Ensure ratios sum to 1
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"Warning: Ratios sum to {total_ratio}, normalizing...")
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio
    
    # Shuffle images
    random.shuffle(images)
    
    total_images = len(images)
    train_end = int(total_images * train_ratio)
    val_end = train_end + int(total_images * val_ratio)
    
    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]
    
    print(f"Dataset split:")
    print(f"  Train: {len(train_images)} images")
    print(f"  Valid: {len(val_images)} images")
    print(f"  Test: {len(test_images)} images")
    
    # Create image ID sets for each split
    train_img_ids = {img['id'] for img in train_images}
    val_img_ids = {img['id'] for img in val_images}
    test_img_ids = {img['id'] for img in test_images}
    
    # Group annotations by image ID
    annotations_by_img = defaultdict(list)
    for ann in annotations:
        annotations_by_img[ann['image_id']].append(ann)
    
    # Split annotations
    train_annotations = []
    val_annotations = []
    test_annotations = []
    
    for img_id, anns in annotations_by_img.items():
        if img_id in train_img_ids:
            train_annotations.extend(anns)
        elif img_id in val_img_ids:
            val_annotations.extend(anns)
        elif img_id in test_img_ids:
            test_annotations.extend(anns)
    
    print(f"Annotations split:")
    print(f"  Train: {len(train_annotations)} annotations")
    print(f"  Valid: {len(val_annotations)} annotations")
    print(f"  Test: {len(test_annotations)} annotations")
    
    return {
        'train': {'images': train_images, 'annotations': train_annotations},
        'valid': {'images': val_images, 'annotations': val_annotations},
        'test': {'images': test_images, 'annotations': test_annotations}
    }

def create_output_structure(output_dir: str, splits: Dict, categories: List[Dict], 
                          image_paths: Dict[str, str], data_dir: str):
    """Create the output directory structure and copy files."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for split_name, split_data in splits.items():
        print(f"\nProcessing {split_name} split...")
        
        # Create split directory
        split_dir = output_path / split_name
        split_dir.mkdir(exist_ok=True)
        
        # Copy images
        copied_images = []
        for img_info in split_data['images']:
            img_filename = img_info['file_name']
            
            if img_filename in image_paths:
                src_path = image_paths[img_filename]
                # Create a simple filename for the destination (just the image name)
                dst_filename = Path(img_filename).name
                dst_path = split_dir / dst_filename
                
                try:
                    # Create parent directory if it doesn't exist
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_path, dst_path)
                    
                    # Update the image info to use the new simple filename
                    img_info_copy = img_info.copy()
                    img_info_copy['file_name'] = dst_filename
                    copied_images.append(img_info_copy)
                    
                    if len(copied_images) <= 5:  # Print first 5 for debugging
                        print(f"  Copied: {img_filename} -> {dst_filename}")
                except Exception as e:
                    print(f"  Error copying {img_filename}: {e}")
            else:
                print(f"  Warning: Image {img_filename} not found, skipping")
        
        print(f"  Successfully copied {len(copied_images)} images")
        
        # Create annotation file
        coco_annotation = {
            'images': copied_images,
            'annotations': split_data['annotations'],
            'categories': categories,
            'info': {
                'description': f'RF-DETR {split_name} dataset',
                'version': '1.0',
                'year': 2025,
                'contributor': 'RF-DETR conversion script'
            }
        }
        
        # Save annotation file
        ann_file = split_dir / '_annotations.coco.json'
        with open(ann_file, 'w') as f:
            json.dump(coco_annotation, f, indent=2)
        
        print(f"  Created annotation file: {ann_file}")
        print(f"  {split_name} split: {len(copied_images)} images, {len(split_data['annotations'])} annotations")

def main():
    parser = argparse.ArgumentParser(description='Convert COCO dataset to RF-DETR format')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to the dataset directory')
    parser.add_argument('--annotation_path', type=str, required=True,
                       help='Path to the COCO annotation JSON file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for converted dataset')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Ratio for training set (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                       help='Ratio for validation set (default: 0.2)')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                       help='Ratio for test set (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible splits (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducible splits
    random.seed(args.seed)
    
    # Validate input paths
    if not os.path.exists(args.data_dir):
        raise ValueError(f"Data directory does not exist: {args.data_dir}")
    
    if not os.path.exists(args.annotation_path):
        raise ValueError(f"Annotation file does not exist: {args.annotation_path}")
    
    print("=== RF-DETR Dataset Conversion ===")
    print(f"Data directory: {args.data_dir}")
    print(f"Annotation file: {args.annotation_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Split ratios - Train: {args.train_ratio}, Val: {args.val_ratio}, Test: {args.test_ratio}")
    
    # Load COCO annotation
    coco_data = load_coco_annotation(args.annotation_path)
    
    # Create category mapping
    category_map = create_category_mapping(coco_data['categories'])
    
    # Find image files
    image_filenames = [img['file_name'] for img in coco_data['images']]
    image_paths = find_image_files(args.data_dir, image_filenames)
    
    # Split dataset
    splits = split_dataset(coco_data['images'], coco_data['annotations'],
                          args.train_ratio, args.val_ratio, args.test_ratio)
    
    # Create output structure
    create_output_structure(args.output_dir, splits, coco_data['categories'], 
                          image_paths, args.data_dir)
    
    print("\n=== Conversion Complete ===")
    print(f"Dataset converted successfully to: {args.output_dir}")
    print("\nDataset structure:")
    print(f"{args.output_dir}/")
    print("├── train/")
    print("│   ├── _annotations.coco.json")
    print("│   └── images...")
    print("├── valid/")
    print("│   ├── _annotations.coco.json")
    print("│   └── images...")
    print("└── test/")
    print("    ├── _annotations.coco.json")
    print("    └── images...")

if __name__ == '__main__':
    main() 