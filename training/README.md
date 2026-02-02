# ASL Hand Sign Model Training

This directory contains the training pipeline for the ASL hand sign classifier.

## Training Results

### Neural Network Model (Landmark-based)
- **Test Accuracy: 100%** ✅
- Training Samples: 27,964 (augmented from 6,991)
- Test Samples: 1,452
- Classes: 27 (A-Z + space)
- Model Type: Dense Neural Network (90 features → 128 → 64 → 32 → 27)

## Requirements

Install the required packages:

```bash
pip install mediapipe opencv-python numpy scikit-learn tensorflow
```

## Training Scripts

### 1. train_cnn_model.py (Recommended)
Trains a CNN on landmark images directly. Uses all images in your dataset.

```bash
python train_cnn_model.py
```

### 2. train_asl_model.py
Trains a dense neural network on MediaPipe-extracted landmarks.

```bash
python train_asl_model.py
```

## Output Files

After training, the following files will be generated in the `output/` directory:

- `best_cnn_model.keras` - Best CNN model (100% accuracy)
- `asl_trained_weights.json` - Exported weights for JavaScript inference
- `cnn_model_config.json` - Model configuration
- `label_encoder.pkl` - Label encoder for class mapping

## Integration

The trained weights (`asl_trained_weights.json`) are automatically copied to `public/` folder for use by the web app. The neural network classifier in `services/nnClassifier.ts` loads these weights dynamically.

## Dataset Structure

The dataset should be organized as:
```
aslhandsignwithlandmark/
└── own_dataset/
    ├── A/
    │   ├── image1.jpg
    │   └── ...
    ├── B/
    │   └── ...
    ├── Z/
    │   └── ...
    └── space/
        └── ...
```

## Features Used (90 total)

1. **Raw landmarks** (63 features) - 21 landmarks × 3 coordinates (x, y, z)
2. **Computed features** (27 features):
   - Finger tip distances from wrist (5)
   - Finger curl angles (5)
   - Finger spread distances (4)
   - Thumb-finger touch detection (4)
   - Palm orientation (3)
   - Fingertip heights (5)
   - Index-middle cross detection (1)
