#!/usr/bin/env python3
"""
Inference Script for Polyp Detection
Supports single image, folder, and video inference with overlay visualization.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image


def draw_boxes(image: np.ndarray,
               boxes: np.ndarray,
               scores: np.ndarray,
               class_names: List[str] = None,
               color: tuple = (0, 255, 0),
               thickness: int = 2) -> np.ndarray:
    """
    Draw bounding boxes on image.
    
    Args:
        image: Input image (RGB)
        boxes: Boxes in [x1, y1, x2, y2] format
        scores: Confidence scores
        class_names: Class names
        color: Box color (RGB)
        thickness: Box thickness
    
    Returns:
        Image with boxes drawn
    """
    img_copy = image.copy()
    
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, box)
        
        # Draw rectangle
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label
        label = f"Polyp: {score:.2f}"
        
        # Calculate label size
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # Draw label background
        cv2.rectangle(
            img_copy,
            (x1, y1 - label_h - baseline - 5),
            (x1 + label_w, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            img_copy,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )
    
    return img_copy


def inference_single_image(model,
                          image_path: Path,
                          output_dir: Path,
                          conf_threshold: float = 0.25,
                          iou_threshold: float = 0.45,
                          use_fp_filter: bool = False,
                          fp_filter = None) -> dict:
    """
    Run inference on a single image.
    
    Returns:
        Dict with inference results
    """
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Failed to load image {image_path}")
        return None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Run inference
    results = model.predict(
        str(image_path),
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False
    )
    
    # Extract predictions
    boxes = []
    scores = []
    
    if len(results) > 0:
        result = results[0]
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes_obj = result.boxes
            if len(boxes_obj) > 0:
                boxes = boxes_obj.xyxy.cpu().numpy()
                scores = boxes_obj.conf.cpu().numpy()
    
    # Apply FP filter if requested
    if use_fp_filter and fp_filter is not None and len(boxes) > 0:
        boxes, scores = fp_filter(img_rgb, boxes, scores)
    
    # Draw boxes
    if len(boxes) > 0:
        img_with_boxes = draw_boxes(img_rgb, boxes, scores)
    else:
        img_with_boxes = img_rgb
    
    # Save output
    output_path = output_dir / f"{image_path.stem}_detected.jpg"
    img_bgr = cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), img_bgr)
    
    return {
        'image_path': str(image_path),
        'num_detections': len(boxes),
        'boxes': boxes.tolist() if len(boxes) > 0 else [],
        'scores': scores.tolist() if len(scores) > 0 else [],
        'output_path': str(output_path)
    }


def inference_folder(model,
                    input_dir: Path,
                    output_dir: Path,
                    conf_threshold: float = 0.25,
                    iou_threshold: float = 0.45,
                    use_fp_filter: bool = False,
                    fp_filter = None) -> List[dict]:
    """
    Run inference on all images in a folder.
    
    Returns:
        List of inference results
    """
    # Get all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(input_dir.glob(f"*{ext}")))
        image_files.extend(list(input_dir.glob(f"*{ext.upper()}")))
    
    print(f"Found {len(image_files)} images in {input_dir}")
    
    results = []
    
    for img_path in tqdm(image_files, desc="Processing images"):
        result = inference_single_image(
            model=model,
            image_path=img_path,
            output_dir=output_dir,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            use_fp_filter=use_fp_filter,
            fp_filter=fp_filter
        )
        
        if result:
            results.append(result)
    
    return results


def inference_video(model,
                   video_path: Path,
                   output_path: Path,
                   conf_threshold: float = 0.25,
                   iou_threshold: float = 0.45,
                   fps: Optional[float] = None,
                   use_fp_filter: bool = False,
                   fp_filter = None) -> dict:
    """
    Run inference on a video and create output video with overlays.
    
    Returns:
        Dict with video inference results
    """
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return None
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Use original FPS if not specified
    if fps is None:
        fps = video_fps
    
    print(f"Video: {total_frames} frames, {video_fps} fps, {width}x{height}")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    frame_results = []
    frame_idx = 0
    
    with tqdm(total=total_frames, desc="Processing video") as pbar:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run inference
            results = model.predict(
                frame_rgb,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False
            )
            
            # Extract predictions
            boxes = []
            scores = []
            
            if len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes_obj = result.boxes
                    if len(boxes_obj) > 0:
                        boxes = boxes_obj.xyxy.cpu().numpy()
                        scores = boxes_obj.conf.cpu().numpy()
            
            # Apply FP filter if requested
            if use_fp_filter and fp_filter is not None and len(boxes) > 0:
                boxes, scores = fp_filter(frame_rgb, boxes, scores)
            
            # Draw boxes
            if len(boxes) > 0:
                frame_with_boxes = draw_boxes(frame_rgb, boxes, scores)
            else:
                frame_with_boxes = frame_rgb
            
            # Write frame
            frame_bgr = cv2.cvtColor(frame_with_boxes, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
            
            # Store results
            frame_results.append({
                'frame': frame_idx,
                'num_detections': len(boxes),
                'boxes': boxes.tolist() if len(boxes) > 0 else [],
                'scores': scores.tolist() if len(scores) > 0 else []
            })
            
            frame_idx += 1
            pbar.update(1)
    
    cap.release()
    out.release()
    
    # Summary stats
    total_detections = sum(r['num_detections'] for r in frame_results)
    frames_with_detections = sum(1 for r in frame_results if r['num_detections'] > 0)
    
    return {
        'video_path': str(video_path),
        'output_path': str(output_path),
        'total_frames': total_frames,
        'frames_with_detections': frames_with_detections,
        'total_detections': total_detections,
        'frame_results': frame_results
    }


def main():
    parser = argparse.ArgumentParser(description="Run polyp detection inference")
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image, folder, or video')
    parser.add_argument('--out', type=str, default='demo_outputs',
                       help='Output directory')
    
    parser.add_argument('--conf-threshold', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou-threshold', type=float, default=0.45,
                       help='IoU threshold for NMS')
    
    parser.add_argument('--fp-filter', type=str, default=None,
                       help='Path to FP filter checkpoint (optional)')
    
    parser.add_argument('--video-fps', type=float, default=None,
                       help='Output video FPS (default: same as input)')
    
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cpu, cuda, or auto)')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    from models.detector import YOLOv8PolypDetector
    
    detector = YOLOv8PolypDetector(device=args.device)
    detector.load_checkpoint(args.checkpoint)
    
    # Load FP filter if provided
    fp_filter = None
    if args.fp_filter:
        print(f"Loading FP filter from {args.fp_filter}")
        from models.secondary_fp_filter import FalsePositiveFilter
        fp_filter = FalsePositiveFilter(
            checkpoint_path=args.fp_filter,
            device=args.device
        )
    
    # Create output directory
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine input type
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: Input path {input_path} does not exist")
        return 1
    
    results = None
    
    if input_path.is_file():
        # Check if video
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        if input_path.suffix.lower() in video_extensions:
            print(f"Running inference on video: {input_path}")
            output_path = output_dir / f"{input_path.stem}_detected.mp4"
            results = inference_video(
                model=detector,
                video_path=input_path,
                output_path=output_path,
                conf_threshold=args.conf_threshold,
                iou_threshold=args.iou_threshold,
                fps=args.video_fps,
                use_fp_filter=args.fp_filter is not None,
                fp_filter=fp_filter
            )
            
            if results:
                print(f"\n✓ Video inference complete:")
                print(f"  Frames processed: {results['total_frames']}")
                print(f"  Frames with detections: {results['frames_with_detections']}")
                print(f"  Total detections: {results['total_detections']}")
                print(f"  Output saved to: {results['output_path']}")
        else:
            # Single image
            print(f"Running inference on image: {input_path}")
            results = inference_single_image(
                model=detector,
                image_path=input_path,
                output_dir=output_dir,
                conf_threshold=args.conf_threshold,
                iou_threshold=args.iou_threshold,
                use_fp_filter=args.fp_filter is not None,
                fp_filter=fp_filter
            )
            
            if results:
                print(f"\n✓ Image inference complete:")
                print(f"  Detections: {results['num_detections']}")
                print(f"  Output saved to: {results['output_path']}")
    
    elif input_path.is_dir():
        # Folder of images
        print(f"Running inference on folder: {input_path}")
        results = inference_folder(
            model=detector,
            input_dir=input_path,
            output_dir=output_dir,
            conf_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold,
            use_fp_filter=args.fp_filter is not None,
            fp_filter=fp_filter
        )
        
        if results:
            total_detections = sum(r['num_detections'] for r in results)
            images_with_detections = sum(1 for r in results if r['num_detections'] > 0)
            
            print(f"\n✓ Folder inference complete:")
            print(f"  Images processed: {len(results)}")
            print(f"  Images with detections: {images_with_detections}")
            print(f"  Total detections: {total_detections}")
            print(f"  Outputs saved to: {output_dir}")
    
    # Save results JSON
    if results:
        import json
        json_path = output_dir / 'inference_results.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved to: {json_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
