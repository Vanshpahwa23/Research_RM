# Polyp Detection Pipeline - Documentation

## Overview

This repository contains a complete, production-quality deep learning pipeline for polyp detection in colonoscopy images and videos. The pipeline includes:

- Dataset acquisition (with synthetic data generation fallback)
- Preprocessing and augmentation
- YOLOv8-based object detection
- Secondary false-positive filter
- Training with 5-fold cross-validation support
- Comprehensive evaluation metrics
- Inference on images and videos

## Quick Start

### Local Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Vanshpahwa23/Research_RM.git
cd Research_RM
git checkout copilot/full-pipeline
```

2. **Create virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Docker Installation

1. **Build Docker image (GPU-enabled):**
```bash
docker build -t polyp-detection:gpu .
```

2. **Build Docker image (CPU-only):**
   
   First, modify Dockerfile to use Python base image instead of CUDA:
```dockerfile
FROM python:3.10-slim
```

Then build:
```bash
docker build -t polyp-detection:cpu .
```

3. **Run container:**
```bash
# GPU version
docker run --gpus all -it -v $(pwd)/data:/app/data polyp-detection:gpu

# CPU version
docker run -it -v $(pwd)/data:/app/data polyp-detection:cpu
```

## Pipeline Execution

### Step 1: Dataset Acquisition

Fetch datasets (attempts real datasets, falls back to synthetic):

```bash
python datasets/fetch_datasets.py \
    --dest data/ \
    --num-train 1000 \
    --num-val 200 \
    --num-test 200
```

**Options:**
- `--dest`: Destination directory for datasets
- `--synthetic-only`: Skip download attempts and generate synthetic data
- `--num-train`, `--num-val`, `--num-test`: Number of synthetic images per split

**Expected output:**
- `data/train/images/` and `data/train/labels/`
- `data/val/images/` and `data/val/labels/`
- `data/test/images/` and `data/test/labels/`
- `data/dataset_manifest.json`

### Step 2: Preprocessing

Preprocess images (resize, CLAHE enhancement, denoising):

```bash
python scripts/data_preprocessor.py \
    --src data/ \
    --dst data_preprocessed/ \
    --img-size 640 480 \
    --clahe \
    --denoise
```

**Options:**
- `--src`: Source directory with raw data
- `--dst`: Destination directory for preprocessed data
- `--img-size WIDTH HEIGHT`: Target image size (default: 640 480)
- `--clahe`: Apply CLAHE contrast enhancement (default: enabled)
- `--no-clahe`: Disable CLAHE
- `--denoise`: Apply Gaussian denoising

### Step 3: Training

Train the YOLOv8 polyp detector:

**Quick demo (2-3 epochs):**
```bash
python train.py \
    --dataset data_preprocessed \
    --epochs 3 \
    --batch-size 16 \
    --img-size 640 \
    --save-dir logs/demo
```

**Full training (200 epochs):**
```bash
python train.py \
    --dataset data_preprocessed \
    --epochs 200 \
    --batch-size 16 \
    --img-size 640 \
    --model-size n \
    --lr 0.01 \
    --optimizer AdamW \
    --patience 50 \
    --amp \
    --save-dir logs/train
```

**5-fold cross-validation:**
```bash
python train.py \
    --dataset data_preprocessed \
    --epochs 200 \
    --batch-size 16 \
    --kfold 5 \
    --save-dir logs/kfold
```

**Options:**
- `--dataset`: Path to preprocessed dataset
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size
- `--img-size`: Input image size
- `--model-size`: YOLOv8 size (n, s, m, l, x)
- `--pretrained`: Use pretrained COCO weights (default: True)
- `--lr`: Learning rate
- `--optimizer`: Optimizer (SGD, Adam, AdamW)
- `--patience`: Early stopping patience
- `--amp`: Use mixed precision training
- `--kfold`: Number of folds for cross-validation (0 for no CV)
- `--device`: Device (cpu, cuda, auto)

**Output:**
- Model checkpoints in `logs/train/weights/`
- Training logs and metrics
- TensorBoard logs

### Step 4: Evaluation

Evaluate the trained model:

```bash
python evaluate.py \
    --checkpoint logs/train/weights/best.pt \
    --test-dir data_preprocessed/test \
    --out results/ \
    --conf-threshold 0.25 \
    --iou-threshold 0.45 \
    --latency-runs 100
```

**Options:**
- `--checkpoint`: Path to model checkpoint
- `--test-dir`: Path to test data directory
- `--out`: Output directory for results
- `--conf-threshold`: Confidence threshold
- `--iou-threshold`: IoU threshold for NMS
- `--latency-runs`: Number of runs for latency measurement

**Output:**
- `results/metrics_demo.csv`: Metrics in CSV format
- `results/evaluation_results.json`: Full results in JSON
- `demo_outputs/eval_overlay_*.jpg`: Sample visualizations

### Step 5: Inference

Run inference on images, folders, or videos:

**Single image:**
```bash
python infer.py \
    --checkpoint logs/train/weights/best.pt \
    --input path/to/image.jpg \
    --out demo_outputs/
```

**Folder of images:**
```bash
python infer.py \
    --checkpoint logs/train/weights/best.pt \
    --input data_preprocessed/val/images \
    --out demo_outputs/
```

**Video:**
```bash
python infer.py \
    --checkpoint logs/train/weights/best.pt \
    --input path/to/video.mp4 \
    --out demo_outputs/ \
    --video-fps 30
```

**With false-positive filter:**
```bash
python infer.py \
    --checkpoint logs/train/weights/best.pt \
    --input path/to/image.jpg \
    --out demo_outputs/ \
    --fp-filter logs/fp_filter/best_fp_filter.pt
```

**Options:**
- `--checkpoint`: Path to model checkpoint
- `--input`: Input image, folder, or video
- `--out`: Output directory
- `--conf-threshold`: Confidence threshold
- `--iou-threshold`: IoU threshold for NMS
- `--fp-filter`: Path to FP filter checkpoint (optional)
- `--video-fps`: Output video FPS

## Testing

Run unit tests:

```bash
pytest tests/ -v
```

Run tests with coverage:

```bash
pytest tests/ -v --cov=. --cov-report=html
```

## TensorBoard Monitoring

Monitor training progress:

```bash
tensorboard --logdir logs/train
```

Then open http://localhost:6006 in your browser.

## Project Structure

```
Research_RM/
├── datasets/
│   └── fetch_datasets.py          # Dataset acquisition
├── scripts/
│   └── data_preprocessor.py       # Data preprocessing
├── utils/
│   └── augmentations.py           # Augmentation utilities
├── models/
│   ├── detector.py                # YOLOv8 detector
│   └── secondary_fp_filter.py     # False positive filter
├── tests/
│   ├── test_preprocessor.py       # Preprocessor tests
│   └── test_model_init.py         # Model initialization tests
├── .github/
│   └── workflows/
│       └── ci.yml                 # CI/CD configuration
├── train.py                       # Training script
├── evaluate.py                    # Evaluation script
├── infer.py                       # Inference script
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker configuration
├── docs/
│   └── README_PIPELINE.md         # This file
└── tasks.md                       # Task tracking and logs
```

## Metrics and Performance

### Detection Metrics
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5
- **mAP@0.75**: Mean Average Precision at IoU threshold 0.75

### Clinical Metrics
- **Sensitivity**: Same as Recall (target ≥95%)
- **Specificity**: True negatives / (True negatives + False positives) (target ≥90%)
- **PPV** (Positive Predictive Value): Same as Precision
- **NPV** (Negative Predictive Value): True negatives / (True negatives + False negatives)

### Inference Latency
- Measured over 100+ inference runs
- Reports mean, std, min, max, median, p95, p99

## Reproducibility

The pipeline ensures reproducibility through:

1. **Fixed random seeds**: Set in `train.py` for numpy, torch, and Python random
2. **Deterministic CUDA operations**: Enabled when available
3. **Version pinning**: All dependencies in `requirements.txt` have minimum versions
4. **Configuration saving**: Training arguments saved to `train_args.json`
5. **Dataset manifest**: `dataset_manifest.json` tracks data sources and counts

## GPU vs CPU

**GPU Training (Recommended):**
- Significantly faster training (10-100x speedup)
- Enable with `--device cuda` or `--device auto`
- Requires CUDA-compatible GPU and drivers

**CPU Training (Fallback):**
- Works on any system
- Slower but functional
- Use `--device cpu`
- Reduce batch size if memory limited

## Troubleshooting

### Issue: ultralytics not found
**Solution:** Install ultralytics: `pip install ultralytics`

### Issue: CUDA out of memory
**Solution:** Reduce batch size: `--batch-size 8` or `--batch-size 4`

### Issue: No test images found
**Solution:** Ensure preprocessing completed successfully and test split exists

### Issue: Training too slow on CPU
**Solution:** 
- Reduce epochs for demo: `--epochs 3`
- Reduce image size: `--img-size 416`
- Use smaller model: `--model-size n`

## Next Steps to Reach Production Targets

### If Sensitivity < 95% or Specificity < 90%:

1. **Collect more real data:**
   - Target: 5,000-10,000 annotated images
   - Include diverse polyp types, sizes, locations
   - Quality annotation with expert review

2. **Hyperparameter tuning:**
   - Use Optuna for systematic search
   - Tune: learning rate, batch size, augmentation strength, model size

3. **Model ensembling:**
   - Combine YOLOv8 with other architectures (EfficientDet, Faster R-CNN)
   - Use voting or weighted averaging

4. **Advanced augmentation:**
   - Test-time augmentation (TTA)
   - Domain-specific augmentations

5. **False-positive filter:**
   - Train secondary classifier on hard negatives
   - Apply during inference

6. **Post-processing:**
   - Adjust confidence thresholds
   - Non-maximum suppression tuning

## Regulatory Considerations

For medical device deployment, consider:

- **FDA SaMD** (Software as a Medical Device) classification
- **IEC 62304**: Medical device software lifecycle
- **ISO 13485**: Quality management systems
- **Clinical validation**: External validation on unseen datasets
- **Bias and fairness**: Test on diverse patient populations
- **Explainability**: Model interpretability for clinicians

## License

[Specify license here]

## Contact

For questions or issues, please open a GitHub issue or contact the repository maintainer.
