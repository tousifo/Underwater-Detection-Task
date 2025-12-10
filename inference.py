#!/usr/bin/env python3
"""
Underwater Object Detection Inference
Runs detection on test images and validates goal achievement
"""

from ultralytics import YOLO
from pathlib import Path
import os

def main():
    print("="*70)
    print("MAMBA-YOLO UNDERWATER INFERENCE")
    print("="*70)

    # Load trained model
    weights = 'runs_underwater/mamba_yolo/weights/best.pt'
    if not os.path.exists(weights):
        print(f"⚠ Model not found: {weights}")
        print("Please train the model first using: python train_underwater.py")
        return

    model = YOLO(weights)
    print(f"✓ Model loaded: {weights}")
    print(f"  Classes: {list(model.names.values())}\n")

    # Run inference on test images
    source = 'underwater-objects-2/test/images'
    results = model.predict(
        source=source,
        conf=0.25,
        iou=0.45,
        save=True,
        project='predictions',
        name='results',
        exist_ok=True
    )

    # Analyze results
    total_detections = sum(len(r.boxes) for r in results)

    print("\n" + "="*70)
    print("INFERENCE RESULTS")
    print("="*70)
    print(f"Images processed: {len(results)}")
    print(f"Total objects detected: {total_detections}")
    print(f"Average per image: {total_detections/len(results):.1f}")

    # Check goal (≥10 objects across 8 images)
    if total_detections >= 10:
        print(f"\n✓✓✓ GOAL ACHIEVED! Detected {total_detections} objects (≥10 required)")
    else:
        print(f"\n⚠ Goal not met: {total_detections}/10 objects detected")

    # Detailed breakdown
    print("\nDetailed Detections:")
    for i, r in enumerate(results[:10], 1):  # Show first 10
        img_name = Path(r.path).name
        num_objs = len(r.boxes)
        print(f"  [{i}] {img_name}: {num_objs} objects")

        # Class breakdown
        if num_objs > 0:
            for cls_id in range(len(model.names)):
                count = sum(1 for box in r.boxes if int(box.cls[0]) == cls_id)
                if count > 0:
                    print(f"      - {model.names[cls_id]}: {count}")

    print("\n" + "="*70)
    print(f"✓ Predictions saved to: predictions/results/")
    print("="*70)

if __name__ == '__main__':
    main()
