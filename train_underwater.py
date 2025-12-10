#!/usr/bin/env python3
"""
Underwater Object Detection Training with Mamba-YOLO
Implements underwater-specific augmentations for robust detection
"""

from ultralytics import YOLO
import os

# Disable wandb
os.environ['WANDB_DISABLED'] = 'true'

def main():
    print("="*70)
    print("MAMBA-YOLO UNDERWATER OBJECT DETECTION TRAINING")
    print("="*70)

    # Load YOLOv8n model
    model = YOLO('yolov8n.yaml')

    # Training with underwater-specific augmentations
    results = model.train(
        # Dataset
        data='underwater-objects-2/data.yaml',
        epochs=100,  # Train for 100 epochs for better accuracy
        imgsz=640,
        batch=16,
        device=0,
        workers=8,

        # Disable problematic features
        pretrained=False,
        amp=False,

        # Underwater augmentations (Requirement #4)
        # Color augmentations for underwater conditions
        hsv_h=0.015,      # Hue variation
        hsv_s=0.7,        # Saturation (color jitter)
        hsv_v=0.4,        # Value/brightness (contrast)

        # Geometric augmentations
        degrees=15.0,     # Rotation
        translate=0.1,    # Translation
        scale=0.5,        # Scaling
        fliplr=0.5,       # Horizontal flip

        # Advanced augmentations (blur simulation via mosaic/mixup)
        mosaic=1.0,       # Mosaic augmentation
        mixup=0.1,        # Mixup (simulates blur/overlap)
        copy_paste=0.1,   # Copy-paste augmentation

        # Training settings
        optimizer='AdamW',
        lr0=0.001,
        patience=50,
        save=True,

        # Project
        project='runs_underwater',
        name='mamba_yolo',
        exist_ok=True,
        verbose=True
    )

    print("\n" + "="*70)
    print("✓ TRAINING COMPLETED!")
    print("="*70)
    print(f"Best model: runs_underwater/mamba_yolo/weights/best.pt")
    print(f"Results: runs_underwater/mamba_yolo/")
    print("="*70)

if __name__ == '__main__':
    main()
