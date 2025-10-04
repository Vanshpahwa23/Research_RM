#!/usr/bin/env python3
"""
Evaluation Script for Polyp Detection
Computes detection metrics, clinical metrics, and inference latency.
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate IoU between two boxes.
    
    Args:
        box1, box2: [x1, y1, x2, y2]
    
    Returns:
        IoU value
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Intersection area
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def yolo_to_xyxy(bbox: List[float], img_width: int, img_height: int) -> np.ndarray:
    """Convert YOLO format to [x1, y1, x2, y2]."""
    x_center, y_center, w, h = bbox
    x1 = (x_center - w/2) * img_width
    y1 = (y_center - h/2) * img_height
    x2 = (x_center + w/2) * img_width
    y2 = (y_center + h/2) * img_height
    return np.array([x1, y1, x2, y2])


def calculate_map(predictions: List[Dict], 
                 ground_truths: List[Dict],
                 iou_threshold: float = 0.5) -> float:
    """
    Calculate mean Average Precision.
    
    Args:
        predictions: List of prediction dicts with 'boxes', 'scores', 'image_id'
        ground_truths: List of GT dicts with 'boxes', 'image_id'
        iou_threshold: IoU threshold
    
    Returns:
        mAP value
    """
    # Sort predictions by score (descending)
    all_preds = []
    for pred in predictions:
        for box, score in zip(pred['boxes'], pred['scores']):
            all_preds.append({
                'image_id': pred['image_id'],
                'box': box,
                'score': score
            })
    
    all_preds = sorted(all_preds, key=lambda x: x['score'], reverse=True)
    
    # Build GT lookup
    gt_lookup = {}
    total_gt = 0
    for gt in ground_truths:
        gt_lookup[gt['image_id']] = gt['boxes']
        total_gt += len(gt['boxes'])
    
    # Calculate precision-recall
    tp = 0
    fp = 0
    precisions = []
    recalls = []
    
    matched_gts = set()
    
    for pred in all_preds:
        image_id = pred['image_id']
        pred_box = pred['box']
        
        gt_boxes = gt_lookup.get(image_id, [])
        
        # Find best matching GT
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt_box in enumerate(gt_boxes):
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        # Check if match
        match_key = (image_id, best_gt_idx)
        if best_iou >= iou_threshold and match_key not in matched_gts:
            tp += 1
            matched_gts.add(match_key)
        else:
            fp += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / total_gt if total_gt > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
    
    # Calculate AP using 11-point interpolation
    ap = 0
    for t in np.linspace(0, 1, 11):
        precisions_at_recall = [p for p, r in zip(precisions, recalls) if r >= t]
        ap += max(precisions_at_recall) if precisions_at_recall else 0
    
    ap /= 11
    
    return ap


def calculate_clinical_metrics(tp: int, fp: int, tn: int, fn: int) -> Dict:
    """
    Calculate clinical metrics.
    
    Returns:
        Dict with sensitivity, specificity, PPV, NPV
    """
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    return {
        'sensitivity': sensitivity * 100,
        'specificity': specificity * 100,
        'ppv': ppv * 100,
        'npv': npv * 100
    }


def measure_inference_latency(model, 
                             test_images: List,
                             num_runs: int = 100,
                             device: str = 'cpu') -> Dict:
    """
    Measure inference latency.
    
    Returns:
        Dict with latency statistics
    """
    print(f"\nMeasuring inference latency over {num_runs} runs...")
    
    latencies = []
    
    # Warmup
    for _ in range(10):
        _ = model.predict(test_images[0], verbose=False)
    
    # Measure
    for _ in tqdm(range(num_runs), desc="Measuring latency"):
        img_idx = np.random.randint(0, len(test_images))
        
        start_time = time.time()
        _ = model.predict(test_images[img_idx], verbose=False)
        end_time = time.time()
        
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)
    
    return {
        'mean_ms': np.mean(latencies),
        'std_ms': np.std(latencies),
        'min_ms': np.min(latencies),
        'max_ms': np.max(latencies),
        'median_ms': np.median(latencies),
        'p95_ms': np.percentile(latencies, 95),
        'p99_ms': np.percentile(latencies, 99)
    }


def evaluate_yolov8(args):
    """Evaluate YOLOv8 model."""
    from models.detector import YOLOv8PolypDetector
    
    print("\n" + "="*80)
    print("EVALUATING YOLOV8 POLYP DETECTOR")
    print("="*80)
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}")
    detector = YOLOv8PolypDetector(device=args.device)
    detector.load_checkpoint(args.checkpoint)
    
    # Get test images
    test_img_dir = Path(args.test_dir) / 'images'
    test_label_dir = Path(args.test_dir) / 'labels'
    
    test_images = list(test_img_dir.glob("*.jpg")) + list(test_img_dir.glob("*.png"))
    print(f"Found {len(test_images)} test images")
    
    if len(test_images) == 0:
        print("Error: No test images found")
        return None
    
    # Run predictions
    print("\nRunning predictions...")
    predictions = []
    ground_truths = []
    
    for img_path in tqdm(test_images, desc="Predicting"):
        # Predict
        results = detector.predict(
            str(img_path),
            conf=args.conf_threshold,
            iou=args.iou_threshold,
            verbose=False
        )
        
        # Extract boxes and scores
        pred_boxes = []
        pred_scores = []
        
        if len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                if len(boxes) > 0:
                    # Convert to numpy
                    pred_boxes = boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                    pred_scores = boxes.conf.cpu().numpy()
        
        predictions.append({
            'image_id': img_path.stem,
            'boxes': pred_boxes,
            'scores': pred_scores
        })
        
        # Load ground truth
        label_path = test_label_dir / (img_path.stem + ".txt")
        gt_boxes = []
        
        if label_path.exists():
            img = cv2.imread(str(img_path))
            img_h, img_w = img.shape[:2]
            
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        bbox = [float(x) for x in parts[1:5]]
                        bbox_xyxy = yolo_to_xyxy(bbox, img_w, img_h)
                        gt_boxes.append(bbox_xyxy)
        
        ground_truths.append({
            'image_id': img_path.stem,
            'boxes': gt_boxes
        })
    
    # Calculate metrics
    print("\nCalculating metrics...")
    
    # mAP@0.5 and mAP@0.75
    map_50 = calculate_map(predictions, ground_truths, iou_threshold=0.5)
    map_75 = calculate_map(predictions, ground_truths, iou_threshold=0.75)
    
    # Detection metrics (per-image level for simplicity)
    tp, fp, tn, fn = 0, 0, 0, 0
    
    for pred, gt in zip(predictions, ground_truths):
        has_pred = len(pred['boxes']) > 0
        has_gt = len(gt['boxes']) > 0
        
        if has_gt and has_pred:
            tp += 1
        elif has_gt and not has_pred:
            fn += 1
        elif not has_gt and has_pred:
            fp += 1
        else:
            tn += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Clinical metrics
    clinical_metrics = calculate_clinical_metrics(tp, fp, tn, fn)
    
    # Inference latency
    latency_metrics = measure_inference_latency(
        detector,
        test_images,
        num_runs=args.latency_runs,
        device=args.device
    )
    
    # Compile results
    results = {
        'detection_metrics': {
            'precision': precision * 100,
            'recall': recall * 100,
            'f1_score': f1 * 100,
            'mAP@0.5': map_50 * 100,
            'mAP@0.75': map_75 * 100,
        },
        'clinical_metrics': clinical_metrics,
        'latency_metrics': latency_metrics,
        'confusion_matrix': {
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn)
        }
    }
    
    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    print("\nDetection Metrics:")
    for key, value in results['detection_metrics'].items():
        print(f"  {key}: {value:.2f}%")
    
    print("\nClinical Metrics:")
    for key, value in results['clinical_metrics'].items():
        print(f"  {key.upper()}: {value:.2f}%")
    
    print("\nInference Latency:")
    for key, value in results['latency_metrics'].items():
        print(f"  {key}: {value:.2f}")
    
    print("\nConfusion Matrix:")
    print(f"  TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    
    # Save results
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics CSV
    metrics_csv_path = output_dir / 'metrics_demo.csv'
    metrics_df = pd.DataFrame([{
        **results['detection_metrics'],
        **results['clinical_metrics'],
        **{f'latency_{k}': v for k, v in results['latency_metrics'].items()},
        **results['confusion_matrix']
    }])
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"\n✓ Saved metrics to {metrics_csv_path}")
    
    # Save JSON results
    json_path = output_dir / 'evaluation_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved results to {json_path}")
    
    # Save overlay images (first 5)
    print("\nSaving overlay images...")
    demo_output_dir = Path('demo_outputs')
    demo_output_dir.mkdir(exist_ok=True)
    
    for i, (img_path, pred) in enumerate(zip(test_images[:5], predictions[:5])):
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Draw predictions
        for box, score in zip(pred['boxes'], pred['scores']):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_rgb, f"{score:.2f}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save
        save_path = demo_output_dir / f"eval_overlay_{i+1}.jpg"
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(save_path), img_bgr)
    
    print(f"✓ Saved {min(5, len(test_images))} overlay images to {demo_output_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate polyp detection model")
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--test-dir', type=str, default='data_preprocessed/test',
                       help='Path to test data directory')
    parser.add_argument('--out', type=str, default='results',
                       help='Output directory for results')
    
    parser.add_argument('--conf-threshold', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou-threshold', type=float, default=0.45,
                       help='IoU threshold for NMS')
    parser.add_argument('--latency-runs', type=int, default=100,
                       help='Number of runs for latency measurement')
    
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cpu, cuda, or auto)')
    
    args = parser.parse_args()
    
    results = evaluate_yolov8(args)
    
    if results:
        print("\n✓ Evaluation complete!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
