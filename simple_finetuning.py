#!/usr/bin/env python3
"""
Simple fine-tuning script for RF-DETR.
Basic code for fine-tuning as specified in the instruction document.
"""

from rfdetr import RFDETRBase

# Dataset path (update this to your converted dataset path)
DATASET_PATH = "/media/vuthede/Lexar/data/oms_incabin/dataset-rf-detr"
OUTPUT_PATH = "./output_nopretrain"

# Initialize model without pretrained weights
model = RFDETRBase(num_classes=10)

# Fine-tune the model
model.train(
    dataset_dir=DATASET_PATH, 
    epochs=10, 
    batch_size=1, 
    grad_accum_steps=1, 
    lr=1e-4, 
    output_dir=OUTPUT_PATH
)

print("Fine-tuning completed!")
print(f"Model saved to: {OUTPUT_PATH}") 