"""
Train ASL classifier using MediaPipe hand landmarks
Uses augmentation to increase training data from reference images
Target: High accuracy for real-time inference
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
import random

# Configuration
DATASET_PATHS = [
    r"D:\rohan minor project\signbridgeupdate\aslhandsignwithlandmark\own_dataset",
    r"D:\rohan minor project\signbridgeupdate\a-zwithspaceimg",
]
OUTPUT_PATH = r"D:\rohan minor project\signbridgeupdate\public"
MODEL_PATH = r"D:\rohan minor project\signbridgeupdate\training\hand_landmarker.task"
RANDOM_SEED = 42
TEST_SPLIT = 0.2
AUGMENTATIONS_PER_IMAGE = 50  # For single reference images

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

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

def extract_landmarks(image):
    """Extract 21 hand landmarks from image"""
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

def augment_image(image):
    """Apply augmentation to image"""
    h, w = image.shape[:2]
    augmented = image.copy()
    
    # Random brightness
    brightness = random.uniform(0.7, 1.3)
    augmented = cv2.convertScaleAbs(augmented, alpha=brightness, beta=0)
    
    # Random contrast
    contrast = random.uniform(0.8, 1.2)
    augmented = cv2.convertScaleAbs(augmented, alpha=contrast, beta=0)
    
    # Random rotation (small angle)
    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    augmented = cv2.warpAffine(augmented, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    # Random scale
    scale = random.uniform(0.85, 1.15)
    new_w, new_h = int(w * scale), int(h * scale)
    augmented = cv2.resize(augmented, (new_w, new_h))
    
    # Crop or pad to original size
    if scale > 1:
        # Crop center
        start_x = (new_w - w) // 2
        start_y = (new_h - h) // 2
        augmented = augmented[start_y:start_y+h, start_x:start_x+w]
    else:
        # Pad
        pad_x = (w - new_w) // 2
        pad_y = (h - new_h) // 2
        padded = np.zeros((h, w, 3), dtype=np.uint8)
        padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = augmented
        augmented = padded
    
    # Random horizontal flip (mirror - common for sign language)
    if random.random() > 0.5:
        augmented = cv2.flip(augmented, 1)
    
    # Random translation
    tx = random.randint(-20, 20)
    ty = random.randint(-20, 20)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    augmented = cv2.warpAffine(augmented, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    # Add slight noise
    noise = np.random.normal(0, 5, augmented.shape).astype(np.int16)
    augmented = np.clip(augmented.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return augmented

def augment_landmarks(landmarks, noise_level=0.02):
    """Add noise to landmarks for augmentation"""
    landmarks = landmarks.copy()
    noise = np.random.normal(0, noise_level, landmarks.shape)
    return landmarks + noise

def load_dataset():
    """Load dataset from multiple sources"""
    X = []
    y = []
    
    for dataset_path in DATASET_PATHS:
        dataset_path = Path(dataset_path)
        print(f"\nLoading from: {dataset_path}")
        
        # Check if it's a folder structure (with subfolders for each class)
        subfolders = [d for d in dataset_path.iterdir() if d.is_dir()]
        
        if subfolders:
            # Folder structure: each subfolder is a class
            for class_folder in sorted(subfolders):
                class_name = class_folder.name.upper()
                if class_name == 'SPACE':
                    class_name = 'space'
                
                images = list(class_folder.glob("*.png")) + list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.jpeg"))
                
                print(f"  Processing {class_name}: {len(images)} images...", end=" ")
                
                success_count = 0
                for img_path in images:
                    image = cv2.imread(str(img_path))
                    if image is None:
                        continue
                    
                    landmarks = extract_landmarks(image)
                    if landmarks is not None:
                        normalized = normalize_landmarks(landmarks)
                        X.append(normalized)
                        y.append(class_name)
                        success_count += 1
                
                print(f"extracted {success_count}")
        else:
            # Flat structure: each image is a class
            image_files = list(dataset_path.glob("*.jpg")) + list(dataset_path.glob("*.png"))
            print(f"  Found {len(image_files)} reference images")
            
            for img_path in image_files:
                class_name = img_path.stem.upper()
                if class_name == 'SPACE':
                    class_name = 'space'
                
                print(f"  Processing {class_name}...", end=" ")
                
                image = cv2.imread(str(img_path))
                if image is None:
                    print("failed to load")
                    continue
                
                success_count = 0
                
                # Try original
                landmarks = extract_landmarks(image)
                if landmarks is not None:
                    normalized = normalize_landmarks(landmarks)
                    X.append(normalized)
                    y.append(class_name)
                    success_count += 1
                    
                    # Add landmark noise augmentations
                    for _ in range(20):
                        aug_landmarks = augment_landmarks(normalized)
                        X.append(aug_landmarks)
                        y.append(class_name)
                        success_count += 1
                
                # Try augmented versions
                for i in range(AUGMENTATIONS_PER_IMAGE):
                    augmented = augment_image(image)
                    landmarks = extract_landmarks(augmented)
                    
                    if landmarks is not None:
                        normalized = normalize_landmarks(landmarks)
                        X.append(normalized)
                        y.append(class_name)
                        success_count += 1
                
                print(f"extracted {success_count}")
    
    return np.array(X), np.array(y)

def create_model(input_dim, num_classes):
    """Create a neural network for landmark classification"""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        
        # First hidden layer
        layers.Dense(512, kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.4),
        
        # Second hidden layer
        layers.Dense(256, kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.4),
        
        # Third hidden layer
        layers.Dense(128, kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        
        # Fourth hidden layer
        layers.Dense(64),
        layers.Activation('relu'),
        layers.Dropout(0.2),
        
        # Output layer
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
    
    output_file = os.path.join(output_path, "asl_trained_weights.json")
    with open(output_file, 'w') as f:
        json.dump(weights_data, f)
    
    print(f"Weights exported to {output_file}")
    return output_file

def main():
    print("=" * 60)
    print("ASL Landmark Model Training (with Augmentation)")
    print("=" * 60)
    
    # Load dataset
    print("\n1. Loading dataset and extracting landmarks...")
    X, y = load_dataset()
    
    # Get unique classes
    classes = sorted(list(set(y)))
    
    print(f"\nDataset summary:")
    print(f"  Total samples: {len(X)}")
    print(f"  Feature dimension: {X.shape[1]}")
    print(f"  Classes: {len(classes)} ({', '.join(classes)})")
    
    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(classes)
    y_encoded = label_encoder.transform(y)
    
    # Check for classes with too few samples and augment them
    from collections import Counter
    class_counts = Counter(y)
    min_samples = 5  # Minimum samples needed per class
    
    print("\nClass distribution:")
    for cls in classes:
        count = class_counts.get(cls, 0)
        print(f"  {cls}: {count} samples")
        
        # If class has too few samples, duplicate them
        if count < min_samples and count > 0:
            indices = [i for i, label in enumerate(y) if label == cls]
            needed = min_samples - count
            for _ in range(needed):
                idx = random.choice(indices)
                X = np.vstack([X, augment_landmarks(X[idx]).reshape(1, -1)])
                y = np.append(y, cls)
            print(f"    -> Augmented to {min_samples} samples")
    
    # Re-encode after augmentation
    y_encoded = label_encoder.transform(y)
    
    # Split data (without stratify if classes are imbalanced)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=y_encoded
        )
    except ValueError:
        print("\nWarning: Using non-stratified split due to class imbalance")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=TEST_SPLIT, random_state=RANDOM_SEED
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
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=150,
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
