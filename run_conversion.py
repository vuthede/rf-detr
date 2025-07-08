#!/usr/bin/env python3
"""
Quick script to run dataset conversion with your specific parameters.
"""

import subprocess
import sys
import os

def main():
    # Your dataset parameters from the instruction document
    data_dir = "/media/vuthede/Lexar/data/oms_incabin"
    annotation_path = "/media/vuthede/Lexar/data/oms_incabin/annotations/ver110/object_anno_7job.json"
    output_dir = "/media/vuthede/Lexar/data/oms_incabin/dataset-rf-detr"  # You can change this to your preferred output location
    
    # Check if paths exist
    if not os.path.exists(data_dir):
        print(f"Error: Data directory does not exist: {data_dir}")
        print("Please update the data_dir path in this script.")
        sys.exit(1)
    
    if not os.path.exists(annotation_path):
        print(f"Error: Annotation file does not exist: {annotation_path}")
        print("Please update the annotation_path in this script.")
        sys.exit(1)
    
    # Run the conversion
    cmd = [
        "python", "convert_dataset.py",
        "--data_dir", data_dir,
        "--annotation_path", annotation_path,
        "--output_dir", output_dir,
        "--train_ratio", "0.7",
        "--val_ratio", "0.2", 
        "--test_ratio", "0.1",
        "--seed", "42"
    ]
    
    print("Running dataset conversion...")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\nConversion completed successfully!")
        print(f"Dataset is ready at: {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"\nConversion failed with error code: {e.returncode}")
        sys.exit(1)
    except FileNotFoundError:
        print("\nError: convert_dataset.py not found. Make sure it's in the same directory.")
        sys.exit(1)

if __name__ == "__main__":
    main() 