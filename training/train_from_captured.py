"""
Train ASL Model from Web-Captured Dataset

This script processes the JSON dataset captured from the browser-based
capture tool and trains a landmark-based neural network model.

Usage:
1. Run the web app and capture images using the "Capture" tab
2. Download the dataset (JSON file)
3. Run this script: python train_from_captured.py --dataset path/to/dataset.json

The script will:
- Extract landmarks from captured images
- Augment the dataset
- Train a neural network
- Export weights for browser use
"""

import os
import json
import base64
import numpy as np
from pathlib import Path
from io import BytesIO
import cv2
from PIL import Image
import argparse
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime

# Configuration
OUTPUT_PATH = Path(__file__).parent.parent / "public"
MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"
RANDOM_SEED = 42
TEST_SPLIT = 0.2

# Initialize MediaPipe Hand Landmarker
base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3
)
hand_landmarker = vision.HandLandmarker.create_from_options(options)


def load_json_dataset(json_path):
    """Load captured dataset from JSON file"""
    print(f"Loading dataset from: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    images = data.get('images', [])
    print(f"Found {len(images)} images in dataset")
    
    # Group by letter
    by_letter = {}
    for img in images:
        letter = img['letter'].upper()
        if letter not in by_letter:
            by_letter[letter] = []
        by_letter[letter].append(img)
    
    print("Distribution:")
    for letter in sorted(by_letter.keys()):
        print(f"  {letter}: {len(by_letter[letter])} images")
    
    return images, by_letter


def base64_to_image(base64_str):
    """Convert base64 string to numpy array"""
    # Remove data URL prefix if present
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    
    img_data = base64.b64decode(base64_str)
    img = Image.open(BytesIO(img_data))
    return np.array(img)


def extract_landmarks_from_image(image_np):
    """Extract 21 hand landmarks from numpy image"""
    # Convert to RGB if needed
    if len(image_np.shape) == 2:
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif image_np.shape[2] == 4:
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    else:
        image_rgb = image_np
    
    # Create MediaPipe image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    
    # Process with MediaPipe
    results = hand_landmarker.detect(mp_image)
    
    if results.hand_landmarks and len(results.hand_landmarks) > 0:
        hand_landmarks = results.hand_landmarks[0]
        landmarks = []
        for lm in hand_landmarks:
            landmarks.extend([lm.x, lm.y, lm.z])
        return np.array(landmarks)
    
    return None


def use_captured_landmarks(img_data):
    """Use landmarks captured with the image if available"""
    if 'landmarks' not in img_data:
        return None
    
    landmarks = img_data['landmarks']
    if not landmarks or len(landmarks) != 21:
        return None
    
    # Flatten to [x1, y1, z1, x2, y2, z2, ...]
    flat = []
    for lm in landmarks:
        flat.extend(lm)
    
    return np.array(flat)


def normalize_landmarks(landmarks):
    """Normalize landmarks relative to wrist"""
    landmarks = landmarks.reshape(-1, 3)
    
    # Center on wrist (landmark 0)
    wrist = landmarks[0].copy()
    landmarks = landmarks - wrist
    
    # Scale by distance from wrist to middle finger MCP
    scale = np.linalg.norm(landmarks[9])
    if scale > 0:
        landmarks = landmarks / scale
    
    return landmarks.flatten()


def augment_landmarks(landmarks, augment_factor=10):
    """Augment landmarks with small random variations"""
    augmented = [landmarks]
    
    for _ in range(augment_factor - 1):
        # Add small noise
        noise = np.random.normal(0, 0.02, landmarks.shape)
        aug = landmarks + noise
        
        # Random scale
        scale = np.random.uniform(0.9, 1.1)
        aug = aug * scale
        
        # Random rotation (2D in x-y plane)
        angle = np.random.uniform(-15, 15) * np.pi / 180
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        aug_reshaped = aug.reshape(-1, 3)
        for i in range(len(aug_reshaped)):
            x, y = aug_reshaped[i, 0], aug_reshaped[i, 1]
            aug_reshaped[i, 0] = x * cos_a - y * sin_a
            aug_reshaped[i, 1] = x * sin_a + y * cos_a
        
        augmented.append(aug_reshaped.flatten())
    
    return augmented


def process_dataset(images, use_captured=True, augment=True, augment_factor=10):
    """Process all images and extract/augment landmarks"""
    X = []
    y = []
    
    for i, img_data in enumerate(images):
        letter = img_data['letter'].upper()
        
        # Try to use captured landmarks first
        if use_captured and 'landmarks' in img_data:
            landmarks = use_captured_landmarks(img_data)
        else:
            landmarks = None
        
        # Fall back to extracting from image
        if landmarks is None:
            try:
                image_np = base64_to_image(img_data['imageData'])
                landmarks = extract_landmarks_from_image(image_np)
            except Exception as e:
                print(f"  Error processing image {i}: {e}")
                continue
        
        if landmarks is None:
            print(f"  Could not extract landmarks from image {i} ({letter})")
            continue
        
        # Normalize landmarks
        normalized = normalize_landmarks(landmarks)
        
        if augment:
            # Augment data
            augmented = augment_landmarks(normalized, augment_factor)
            for aug in augmented:
                X.append(aug)
                y.append(letter)
        else:
            X.append(normalized)
            y.append(letter)
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(images)} images...")
    
    return np.array(X), np.array(y)


def create_model(input_dim, num_classes):
    """Create neural network for landmark classification"""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def export_to_json(model, classes, output_path):
    """Export model weights to JSON for browser use"""
    weights_data = {
        "classes": classes,
        "architecture": [],
        "weights": []
    }
    
    for i, layer in enumerate(model.layers):
        layer_weights = layer.get_weights()
        if layer_weights:
            layer_data = {
                "name": layer.name,
                "type": layer.__class__.__name__,
                "weights": [w.tolist() for w in layer_weights]
            }
            weights_data["weights"].append(layer_data)
    
    output_file = output_path / "asl_trained_weights.json"
    with open(output_file, 'w') as f:
        json.dump(weights_data, f)
    
    print(f"✅ Exported weights to: {output_file}")
    
    # Also save classes separately
    classes_file = output_path / "asl_classes.json"
    with open(classes_file, 'w') as f:
        json.dump({"classes": classes}, f)
    
    print(f"✅ Exported classes to: {classes_file}")


def train_model(X, y, classes):
    """Train the model"""
    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(classes)
    y_encoded = label_encoder.transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=y_encoded
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Create model
    model = create_model(X.shape[1], len(classes))
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.0001
        )
    ]
    
    # Train
    print("\n🏋️ Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n📊 Final Test Accuracy: {test_acc * 100:.2f}%")
    
    return model, history, test_acc


def main():
    parser = argparse.ArgumentParser(description='Train ASL model from captured dataset')
    parser.add_argument('--dataset', type=str, required=True, help='Path to captured JSON dataset')
    parser.add_argument('--augment', type=int, default=10, help='Augmentation factor (default: 10)')
    parser.add_argument('--use-captured', action='store_true', default=True, help='Use landmarks captured with images')
    parser.add_argument('--output', type=str, default=None, help='Output directory for model weights')
    
    args = parser.parse_args()
    
    output_path = Path(args.output) if args.output else OUTPUT_PATH
    output_path.mkdir(exist_ok=True, parents=True)
    
    print("=" * 60)
    print("ASL Model Training from Captured Dataset")
    print("=" * 60)
    
    # Load dataset
    images, by_letter = load_json_dataset(args.dataset)
    
    if not images:
        print("❌ No images found in dataset!")
        return
    
    # Get classes
    classes = sorted(by_letter.keys())
    print(f"\nClasses: {classes}")
    
    # Process dataset
    print("\n📦 Processing dataset...")
    X, y = process_dataset(
        images, 
        use_captured=args.use_captured, 
        augment=True, 
        augment_factor=args.augment
    )
    
    if len(X) == 0:
        print("❌ No valid samples after processing!")
        return
    
    print(f"\n✅ Total samples after augmentation: {len(X)}")
    
    # Train model
    model, history, accuracy = train_model(X, y, classes)
    
    # Export weights
    print("\n💾 Exporting model...")
    export_to_json(model, classes, output_path)
    
    # Save Keras model too
    model_file = output_path / "asl_landmark_model.keras"
    model.save(model_file)
    print(f"✅ Saved Keras model to: {model_file}")
    
    # Save training info
    info_file = output_path / "training_info.json"
    with open(info_file, 'w') as f:
        json.dump({
            "trained_at": datetime.now().isoformat(),
            "samples": len(X),
            "classes": classes,
            "accuracy": float(accuracy),
            "augmentation_factor": args.augment
        }, f, indent=2)
    print(f"✅ Saved training info to: {info_file}")
    
    print("\n" + "=" * 60)
    print(f"🎉 Training complete! Accuracy: {accuracy * 100:.2f}%")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Refresh your SignBridge app to use the new model")
    print("2. The new weights are in:", output_path / "asl_trained_weights.json")


if __name__ == "__main__":
    main()
