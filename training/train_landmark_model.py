"""
Train ASL classifier using MediaPipe hand landmarks
Target: 85%+ accuracy
"""

import os
import json
import numpy as np
from pathlib import Path
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Configuration
DATASET_PATH = r"D:\rohan minor project\signbridgeupdate\aslhandsignwithlandmark\own_dataset"
OUTPUT_PATH = r"D:\rohan minor project\signbridgeupdate\public"
MODEL_PATH = r"D:\rohan minor project\signbridgeupdate\training\hand_landmarker.task"
RANDOM_SEED = 42
TEST_SPLIT = 0.2

# Initialize MediaPipe Hand Landmarker (Tasks API)
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3
)
hand_landmarker = vision.HandLandmarker.create_from_options(options)

def extract_landmarks(image_path):
    """Extract 21 hand landmarks (63 features: x, y, z for each)"""
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create MediaPipe image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    
    # Process with MediaPipe
    results = hand_landmarker.detect(mp_image)
    
    if results.hand_landmarks and len(results.hand_landmarks) > 0:
        hand_landmarks = results.hand_landmarks[0]
        
        # Extract normalized coordinates
        landmarks = []
        for lm in hand_landmarks:
            landmarks.extend([lm.x, lm.y, lm.z])
        
        return np.array(landmarks)
    
    return None

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

def load_dataset():
    """Load dataset and extract landmarks"""
    X = []
    y = []
    
    dataset_path = Path(DATASET_PATH)
    classes = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])
    
    print(f"Found {len(classes)} classes: {classes}")
    
    for class_name in classes:
        class_path = dataset_path / class_name
        images = list(class_path.glob("*.png")) + list(class_path.glob("*.jpg")) + list(class_path.glob("*.jpeg"))
        
        print(f"Processing {class_name}: {len(images)} images...", end=" ")
        
        success_count = 0
        for img_path in images:
            landmarks = extract_landmarks(img_path)
            if landmarks is not None:
                # Normalize landmarks
                normalized = normalize_landmarks(landmarks)
                X.append(normalized)
                y.append(class_name)
                success_count += 1
        
        print(f"extracted {success_count} landmarks")
    
    return np.array(X), np.array(y), classes

def create_model(input_dim, num_classes):
    """Create a neural network for landmark classification"""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
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
            
            # Store architecture info
            if hasattr(layer, 'units'):
                weights_data["architecture"].append({
                    "name": layer.name,
                    "type": layer.__class__.__name__,
                    "units": layer.units
                })
    
    output_file = os.path.join(output_path, "asl_trained_weights.json")
    with open(output_file, 'w') as f:
        json.dump(weights_data, f)
    
    print(f"Weights exported to {output_file}")
    return output_file

def main():
    print("=" * 60)
    print("ASL Landmark Model Training")
    print("=" * 60)
    
    # Load dataset
    print("\n1. Loading dataset and extracting landmarks...")
    X, y, classes = load_dataset()
    
    print(f"\nDataset summary:")
    print(f"  Total samples: {len(X)}")
    print(f"  Feature dimension: {X.shape[1]}")
    print(f"  Classes: {len(classes)}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=y_encoded
    )
    
    print(f"\n2. Data split:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    # Create and train model
    print("\n3. Training model...")
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
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n4. Evaluating model...")
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\n" + "=" * 60)
    print(f"RESULTS:")
    print(f"  Training Accuracy: {train_acc*100:.2f}%")
    print(f"  Test Accuracy: {test_acc*100:.2f}%")
    print(f"=" * 60)
    
    if test_acc >= 0.85:
        print("✓ Target accuracy of 85% ACHIEVED!")
    else:
        print(f"✗ Target accuracy not met. Need {(0.85-test_acc)*100:.2f}% more.")
    
    # Export weights
    print("\n5. Exporting model weights...")
    export_to_json(model, classes, OUTPUT_PATH)
    
    # Also save keras model
    model.save(os.path.join(OUTPUT_PATH, "asl_landmark_model.keras"))
    print(f"Keras model saved to {OUTPUT_PATH}/asl_landmark_model.keras")
    
    print("\n✓ Training complete!")
    return test_acc

if __name__ == "__main__":
    main()
