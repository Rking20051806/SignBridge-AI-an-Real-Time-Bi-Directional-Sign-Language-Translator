"""
ASL Hand Sign Classifier Training Script
Uses MediaPipe to extract hand landmarks from images and trains a lightweight neural network.
Optimized for fast inference in browser with minimal complexity.
"""

import os
import json
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import urllib.request

# Configuration
DATASET_PATH = Path(__file__).parent.parent / "landmarkingaslaphabetand number" / "processed_combine_asl_dataset"
OUTPUT_PATH = Path(__file__).parent / "output"
CLASSES = list("abcdefghijklmnopqrstuvwxyz") + list("0123456789")  # a-z + 0-9

# MediaPipe hand landmark model URL
HAND_LANDMARKER_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
HAND_LANDMARKER_MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"

def ensure_model_downloaded():
    """Download the hand landmarker model if not present."""
    if not HAND_LANDMARKER_MODEL_PATH.exists():
        print(f"Downloading MediaPipe hand landmarker model...")
        urllib.request.urlretrieve(HAND_LANDMARKER_MODEL_URL, HAND_LANDMARKER_MODEL_PATH)
        print(f"Model downloaded to: {HAND_LANDMARKER_MODEL_PATH}")
    return HAND_LANDMARKER_MODEL_PATH

def extract_landmarks_from_image(image_path, detector):
    """Extract normalized hand landmarks from an image using Tasks API."""
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    
    # Try multiple preprocessing approaches for better detection
    approaches = [
        lambda img: img,  # Original
        lambda img: cv2.resize(img, (640, 640)),  # Resize
        lambda img: cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) if len(img.shape) == 3 else cv2.equalizeHist(img),  # Histogram eq
    ]
    
    for preprocess in approaches:
        try:
            processed = preprocess(image.copy())
            # Ensure 3 channels
            if len(processed.shape) == 2:
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
            elif processed.shape[2] == 3:
                processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=processed)
            results = detector.detect(mp_image)
            
            if results.hand_landmarks and len(results.hand_landmarks) > 0:
                hand_landmarks = results.hand_landmarks[0]
                landmarks = []
                for lm in hand_landmarks:
                    landmarks.extend([lm.x, lm.y, lm.z])
                return np.array(landmarks)
        except:
            continue
    
    return None

def normalize_landmarks(landmarks):
    """
    Normalize landmarks to be translation and scale invariant.
    This improves model generalization.
    """
    if landmarks is None or len(landmarks) != 63:
        return None
    
    # Reshape to (21, 3)
    points = landmarks.reshape(21, 3)
    
    # Center around wrist (landmark 0)
    wrist = points[0].copy()
    points = points - wrist
    
    # Scale by palm width (distance from index MCP to pinky MCP)
    palm_width = np.linalg.norm(points[5] - points[17])
    if palm_width > 0:
        points = points / palm_width
    
    return points.flatten()

def compute_hand_features(landmarks):
    """
    Compute additional geometric features from landmarks.
    These features are specifically designed for ASL recognition.
    """
    if landmarks is None or len(landmarks) != 63:
        return None
    
    points = landmarks.reshape(21, 3)
    
    # Normalize first
    wrist = points[0].copy()
    points = points - wrist
    palm_width = np.linalg.norm(points[5] - points[17])
    if palm_width > 0:
        points = points / palm_width
    
    features = []
    
    # 1. Finger tip distances from wrist (5 features)
    finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
    for tip in finger_tips:
        features.append(np.linalg.norm(points[tip]))
    
    # 2. Finger curl angles (5 features)
    # For each finger, compute angle between MCP-PIP and PIP-TIP vectors
    finger_joints = [
        (1, 2, 4),    # Thumb: CMC, MCP, TIP
        (5, 6, 8),    # Index: MCP, PIP, TIP
        (9, 10, 12),  # Middle: MCP, PIP, TIP
        (13, 14, 16), # Ring: MCP, PIP, TIP
        (17, 18, 20)  # Pinky: MCP, PIP, TIP
    ]
    
    for mcp, pip, tip in finger_joints:
        v1 = points[pip] - points[mcp]
        v2 = points[tip] - points[pip]
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 > 0 and norm2 > 0:
            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            cos_angle = np.clip(cos_angle, -1, 1)
            features.append(cos_angle)  # -1 to 1
        else:
            features.append(0)
    
    # 3. Finger spread (distances between adjacent fingertips) (4 features)
    tip_pairs = [(4, 8), (8, 12), (12, 16), (16, 20)]
    for t1, t2 in tip_pairs:
        features.append(np.linalg.norm(points[t1] - points[t2]))
    
    # 4. Thumb-finger touch detection (4 features)
    thumb_tip = points[4]
    for tip in [8, 12, 16, 20]:
        dist = np.linalg.norm(thumb_tip - points[tip])
        features.append(1.0 if dist < 0.15 else 0.0)
    
    # 5. Palm orientation features (3 features)
    # Normal vector of palm plane using wrist, index MCP, pinky MCP
    v1 = points[5] - points[0]   # Wrist to index MCP
    v2 = points[17] - points[0]  # Wrist to pinky MCP
    palm_normal = np.cross(v1, v2)
    norm = np.linalg.norm(palm_normal)
    if norm > 0:
        palm_normal = palm_normal / norm
    features.extend(palm_normal.tolist())
    
    # 6. Fingertip heights relative to wrist (5 features)
    for tip in finger_tips:
        features.append(points[tip][1])  # Y coordinate (up/down)
    
    # 7. Index-middle finger cross detection for 'R' (1 feature)
    index_middle_dist = np.linalg.norm(points[8] - points[12])
    features.append(1.0 if index_middle_dist < 0.1 else 0.0)
    
    return np.array(features)

def load_dataset():
    """Load images and extract landmarks from the dataset."""
    print("Loading dataset and extracting landmarks...")
    
    # Ensure model is downloaded
    model_path = ensure_model_downloaded()
    
    X_raw = []      # Raw normalized landmarks (63 features)
    X_features = [] # Computed features (27 features)
    y = []          # Labels
    
    # Create hand landmarker using Tasks API with lower thresholds
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.3,  # Lower threshold for detection
        min_tracking_confidence=0.3
    )
    
    with vision.HandLandmarker.create_from_options(options) as detector:
        for class_name in CLASSES:
            class_path = DATASET_PATH / class_name
            
            if not class_path.exists():
                print(f"Warning: Class folder not found: {class_path}")
                continue
            
            image_files = list(class_path.glob("*.jpg")) + list(class_path.glob("*.png"))
            print(f"Processing {class_name}: {len(image_files)} images")
            
            successful = 0
            for img_path in image_files:
                landmarks = extract_landmarks_from_image(img_path, detector)
                
                if landmarks is not None:
                    # Normalize raw landmarks
                    norm_landmarks = normalize_landmarks(landmarks)
                    
                    # Compute additional features
                    features = compute_hand_features(landmarks)
                    
                    if norm_landmarks is not None and features is not None:
                        X_raw.append(norm_landmarks)
                        X_features.append(features)
                        y.append(class_name)
                        successful += 1
            
            print(f"  Successfully extracted: {successful}/{len(image_files)}")
    
    return np.array(X_raw), np.array(X_features), np.array(y)

def build_simple_model(input_dim, num_classes):
    """
    Build a simple but effective neural network for landmark classification.
    Designed for fast inference - can be converted to TensorFlow.js
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(input_dim,)),
        
        # Dense layers with dropout for regularization
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.1),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model():
    """Main training function."""
    # Create output directory
    OUTPUT_PATH.mkdir(exist_ok=True)
    
    # Load data
    X_raw, X_features, y = load_dataset()
    
    if len(X_raw) == 0:
        print("Error: No data loaded. Check dataset path.")
        return
    
    print(f"\nDataset loaded: {len(X_raw)} samples")
    
    # Combine raw landmarks and computed features
    X = np.hstack([X_raw, X_features])
    print(f"Total features per sample: {X.shape[1]} (63 landmarks + {X_features.shape[1]} computed)")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    
    print(f"Classes: {label_encoder.classes_}")
    print(f"Number of classes: {num_classes}")
    
    # Check minimum samples per class for stratified split
    from collections import Counter
    class_counts = Counter(y_encoded)
    min_samples = min(class_counts.values())
    
    # Split data - use stratified only if enough samples per class
    if min_samples >= 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
    else:
        print("Warning: Some classes have too few samples for stratified split. Using random split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Build and train model
    print("\n" + "="*50)
    print("Training Neural Network Model...")
    print("="*50)
    
    import tensorflow as tf
    from tensorflow import keras
    
    model = build_simple_model(X.shape[1], num_classes)
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Per-class accuracy
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print("\nPer-class accuracy:")
    for i, class_name in enumerate(label_encoder.classes_):
        mask = y_test == i
        if mask.sum() > 0:
            class_acc = (y_pred_classes[mask] == y_test[mask]).mean()
            print(f"  {class_name}: {class_acc*100:.1f}%")
    
    # Save model
    print("\n" + "="*50)
    print("Saving Model...")
    print("="*50)
    
    # Save Keras model
    model.save(OUTPUT_PATH / "asl_model.keras")
    print(f"Saved: {OUTPUT_PATH / 'asl_model.keras'}")
    
    # Save label encoder
    with open(OUTPUT_PATH / "label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
    print(f"Saved: {OUTPUT_PATH / 'label_encoder.pkl'}")
    
    # Export model weights and config for JavaScript
    export_for_javascript(model, label_encoder)
    
    print("\n✅ Training complete!")
    
    return model, label_encoder, history

def export_for_javascript(model, label_encoder):
    """
    Export model weights in a format that can be loaded in JavaScript.
    Creates a standalone classifier that doesn't require TensorFlow.js
    """
    print("\nExporting model for JavaScript...")
    
    # Get all weights
    weights = {}
    for i, layer in enumerate(model.layers):
        layer_weights = layer.get_weights()
        if layer_weights:
            weights[f"layer_{i}"] = {
                "name": layer.name,
                "type": layer.__class__.__name__,
                "weights": [w.tolist() for w in layer_weights]
            }
    
    # Export configuration
    config = {
        "input_dim": model.input_shape[1],
        "num_classes": len(label_encoder.classes_),
        "classes": label_encoder.classes_.tolist(),
        "architecture": [
            {"type": "dense", "units": 128, "activation": "relu"},
            {"type": "batchnorm"},
            {"type": "dense", "units": 64, "activation": "relu"},
            {"type": "batchnorm"},
            {"type": "dense", "units": 32, "activation": "relu"},
            {"type": "dense", "units": len(label_encoder.classes_), "activation": "softmax"}
        ]
    }
    
    # Save weights as JSON for JavaScript import
    export_data = {
        "config": config,
        "weights": weights
    }
    
    with open(OUTPUT_PATH / "asl_model_weights.json", "w") as f:
        json.dump(export_data, f)
    print(f"Saved: {OUTPUT_PATH / 'asl_model_weights.json'}")
    
    # Generate TypeScript classifier code
    generate_typescript_classifier(model, label_encoder)

def generate_typescript_classifier(model, label_encoder):
    """Generate an optimized TypeScript classifier using the trained weights."""
    
    # Extract weights from model
    dense_layers = []
    bn_layers = []
    
    for layer in model.layers:
        weights = layer.get_weights()
        if 'dense' in layer.name.lower() and weights:
            dense_layers.append({
                'kernel': weights[0].tolist(),
                'bias': weights[1].tolist()
            })
        elif 'batch_normalization' in layer.name.lower() and weights:
            bn_layers.append({
                'gamma': weights[0].tolist(),
                'beta': weights[1].tolist(),
                'moving_mean': weights[2].tolist(),
                'moving_var': weights[3].tolist()
            })
    
    classes = label_encoder.classes_.tolist()
    
    # Generate compact weights file
    model_data = {
        'dense': dense_layers,
        'bn': bn_layers,
        'classes': classes
    }
    
    with open(OUTPUT_PATH / "asl_trained_weights.json", "w") as f:
        json.dump(model_data, f)
    
    print(f"Saved: {OUTPUT_PATH / 'asl_trained_weights.json'}")
    print("\n📁 Generated files can be used to update the TypeScript classifier.")

if __name__ == "__main__":
    train_model()
