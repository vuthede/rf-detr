#!/usr/bin/env python3
"""
RF-DETR Video Inference Script
Real-time video inference with frame-by-frame visualization.
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image
import supervision as sv
from rfdetr import RFDETRBase

# Configuration
# video_path = "/home/vuthede/OMS_AI/.github/rf-detr/videos/Leopard_250619_6.mp4"  # Set this to your video path
video_path = "/home/vuthede/Downloads/joctech02072025.mp4"
ckpt = "/home/vuthede/OMS_AI/.github/rf-detr/output/checkpoint_best_ema.pth"

# Class names for your dataset (10 classes)
CLASS_NAMES = {
    1: "person",
    2: "head", 
    3: "steering_wheel",
    4: "laptop",
    5: "cell_phone",
    7: "infant",
    8: "baby_seat",
    9: "food",
    10: "cigarette",
    11: "bag"
}

def load_model(checkpoint_path, num_classes=10):
    """Load the trained RF-DETR model from checkpoint."""
    print(f"\n=== MODEL LOADING PROCESS ===")
    print(f"Loading model from: {checkpoint_path}")
    
    # Initialize model
    print("Step 1: Initializing RF-DETR model...")
    print(f"  - Number of classes: {num_classes}")
    model = RFDETRBase(pretrain_weights=checkpoint_path)
    print("✓ Model initialized successfully")
    
    # # Load checkpoint
    # if os.path.exists(checkpoint_path):
    #     print("Step 2: Loading checkpoint...")
    #     try:
    #         print("  - Loading checkpoint file...")
    #         checkpoint = torch.load(checkpoint_path, map_location='cpu')
    #         print(f"  - Checkpoint keys: {list(checkpoint.keys())}")
            
    #         # Handle different checkpoint formats
    #         if 'model' in checkpoint:
    #             print("  - Loading state dict from 'model' key...")
    #             model.model.model.load_state_dict(checkpoint['model'], strict=False)
    #             print("✓ Model loaded from 'model' key")
    #         elif 'ema_model' in checkpoint:
    #             print("  - Loading state dict from 'ema_model' key...")
    #             model.model.model.load_state_dict(checkpoint['ema_model'], strict=False)
    #             print("✓ Model loaded from 'ema_model' key")
    #         else:
    #             print("  - Loading state dict dicheckpointcheckpointrectly from checkpoint...")
    #             model.model.model.load_state_dict(checkpoint, strict=False)
    #             print("✓ Model loaded directly from checkpoint")
                
    #     except Exception as e:
    #         print(f"Warning: Could not load checkpoint: {e}")
    #         print("Using pretrained weights instead")
    # else:
    #     print(f"Warning: Checkpoint not found at {checkpoint_path}")
    #     print("Using pretrained weights instead")
    
    # Check device
    print("Step 3: Checking device availability...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  - PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - CUDA device count: {torch.cuda.device_count()}")
        print(f"  - Current CUDA device: {torch.cuda.current_device()}")
        print(f"  - CUDA device name: {torch.cuda.get_device_name()}")
    print(f"  - Model will run on: {device}")
    
    # Optimize for inference
    print("Step 4: Optimizing model for inference...")
    model.optimize_for_inference()
    print("✓ Model optimization complete")
    
    # Check model device after optimization
    print("Step 5: Checking model device placement...")
    model_device = next(model.model.model.parameters()).device
    print(f"  - Model is on device: {model_device}")
    print("=== MODEL LOADING COMPLETE ===\n")
    
    return model

def inference_video_realtime(model, video_path, output_path, confidence_threshold=0.5):
    """Run inference on video with real-time visualization."""
    print(f"Running real-time inference on video: {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Setup video writer
    print("Setting up video writer...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print(f"✓ Video writer setup complete, output: {output_path}")
    
    # Setup annotators
    annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
    
    frame_count = 0
    
    while True:
        print(f"\n--- Processing Frame {frame_count + 1} ---")
        
        # Step 1: Read frame
        print("Step 1: Reading frame from video...")
        ret, frame = cap.read()
        if not ret:
            print("End of video reached")
            break
        
        frame_count += 1
        print(f"✓ Frame {frame_count} read successfully, shape: {frame.shape}")
        
        # Step 2: Preprocessing
        print("Step 2: Preprocessing frame...")
        print("  - Converting BGR to RGB...")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print("  - Converting to PIL Image...")
        pil_image = Image.fromarray(frame_rgb)
        print(f"✓ Preprocessing complete, image size: {pil_image.size}")
        
        # Step 3: Model inference
        print("Step 3: Running model inference...")
        print(f"  - Model device: {next(model.model.model.parameters()).device}")
        print(f"  - Input image size: {pil_image.size}")
        import time
        start_time = time.time()
        detections = model.predict(pil_image, threshold=confidence_threshold)
        inference_time = time.time() - start_time
        print(f"✓ Model inference complete in {inference_time:.3f}s")
        print(f"  - Found {len(detections)} raw detections above threshold {confidence_threshold}")
        
        # Filter detections to only include your 10 defined classes (1-10)
        # valid_class_mask = (detections.class_id >= 1) & (detections.class_id <= 10)
        # detections = detections[valid_class_mask]
        print(f"  - Filtered to {len(detections)} detections from your 10 classes")
        
        if torch.cuda.is_available():
            print(f"  - GPU memory used: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        
        # Step 4: Post-processing
        print("Step 4: Post-processing detections...")
        print("  - Creating labels with class names and confidence scores...")
        labels = []
        for i, (class_id, confidence) in enumerate(zip(detections.class_id, detections.confidence)):
            class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
            labels.append(f"{class_name} {confidence:.2f}")
            print(f"    Detection {i+1}: {class_name} (confidence: {confidence:.3f})")
        
        print("  - Annotating frame with bounding boxes...")
        annotated_frame = annotator.annotate(scene=frame_rgb, detections=detections)
        print("  - Adding labels to frame...")
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )
        
        print("  - Converting RGB back to BGR for display...")
        display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        print("✓ Post-processing complete")
        
        # Add frame info
        info_text = f"Frame: {frame_count}/{total_frames} | Detections: {len(detections)} | Press 'q' to quit, SPACE to pause"
        cv2.putText(display_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Print detection info every 30 frames
        if frame_count % 30 == 0:
            print(f"Frame {frame_count}/{total_frames} - Found {len(detections)} detections")
            for i, (class_id, confidence, bbox) in enumerate(zip(detections.class_id, detections.confidence, detections.xyxy)):
                class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
                x1, y1, x2, y2 = bbox
                print(f"  {i+1}. {class_name}: {confidence:.3f} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
        
        # Write frame to output video
        out.write(display_frame)
        print(f"✓ Frame {frame_count} written to output video")
    
    cap.release()
    out.release()
    print(f"✓ Output video saved to: {output_path}")
    
    print(f"✓ Video inference completed. Processed {frame_count} frames.")

def main():
    """Main inference function."""
    print("=== RF-DETR Video Inference ===")
    
    # Check if video path is set
    if not video_path or not os.path.exists(video_path):
        print("Error: Please set a valid video_path in the script")
        print("Example: video_path = '/path/to/your/video.mp4'")
        return
    
    # Load model
    model = load_model(ckpt)
    
    # Run video inference with output video
    output_path = "output_video.mp4"  # Output video path
    inference_video_realtime(
        model,
        video_path,
        output_path,
        confidence_threshold=0.6  # Adjust confidence threshold as needed
    )
    
    print("\n=== Inference Complete ===")

if __name__ == "__main__":
    main()
