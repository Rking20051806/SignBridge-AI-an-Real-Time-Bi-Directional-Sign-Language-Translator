"""
ASL Hand Sign CNN Classifier Training Script
Trains directly on landmark images (images with pre-drawn landmarks).
Uses CNN for image classification - works with all images in the dataset.
Target: 92%+ accuracy
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Configuration
DATASET_PATH = Path(__file__).parent.parent / "aslhandsignwithlandmark" / "own_dataset"
OUTPUT_PATH = Path(__file__).parent / "output"
CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["space"]

# Image settings
IMG_SIZE = 64  # Smaller size for faster training and inference
BATCH_SIZE = 32
EPOCHS = 50

def load_and_preprocess_image(image_path, img_size=IMG_SIZE):
    """Load and preprocess a single image."""
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    
    # Convert to grayscale (landmarks are usually colored on dark/white background)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize
    resized = cv2.resize(gray, (img_size, img_size))
    
    # Normalize to 0-1
    normalized = resized.astype(np.float32) / 255.0
    
    return normalized

def load_dataset():
    """Load all images from the dataset."""
    print("Loading dataset...")
    
    X = []
    y = []
    
    for class_name in CLASSES:
        class_path = DATASET_PATH / class_name
        
        if not class_path.exists():
            print(f"Warning: Class folder not found: {class_path}")
            continue
        
        image_files = list(class_path.glob("*.jpg")) + list(class_path.glob("*.png"))
        print(f"Processing {class_name}: {len(image_files)} images", end="")
        
        successful = 0
        for img_path in image_files:
            img = load_and_preprocess_image(img_path)
            if img is not None:
                X.append(img)
                y.append(class_name)
                successful += 1
        
        print(f" -> Loaded: {successful}")
    
    X = np.array(X)
    y = np.array(y)
    
    # Add channel dimension for CNN
    X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    
    return X, y

def build_cnn_model(input_shape, num_classes):
    """
    Build a lightweight CNN for landmark image classification.
    Designed for fast inference and good accuracy.
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    model = keras.Sequential([
        # Input
        layers.Input(shape=input_shape),
        
        # Conv Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Conv Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Conv Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten and Dense
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        
        # Output
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_data_augmentation():
    """Create data augmentation layer for training."""
    import tensorflow as tf
    from tensorflow.keras import layers
    
    return keras.Sequential([
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
    ])

def train_model():
    """Main training function."""
    import tensorflow as tf
    from tensorflow import keras
    
    # Create output directory
    OUTPUT_PATH.mkdir(exist_ok=True)
    
    # Load data
    X, y = load_dataset()
    
    if len(X) == 0:
        print("Error: No data loaded. Check dataset path.")
        return
    
    print(f"\n{'='*50}")
    print(f"Dataset loaded: {len(X)} samples")
    print(f"Image shape: {X.shape[1:]}")
    print(f"{'='*50}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    
    print(f"Classes: {label_encoder.classes_}")
    print(f"Number of classes: {num_classes}")
    
    # Check class distribution
    from collections import Counter
    class_counts = Counter(y)
    print("\nClass distribution:")
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls}: {count}")
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
    )
    
    # Further split training into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Build model
    print(f"\n{'='*50}")
    print("Building CNN Model...")
    print(f"{'='*50}")
    
    model = build_cnn_model(X_train.shape[1:], num_classes)
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            mode='max'
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),
        keras.callbacks.ModelCheckpoint(
            OUTPUT_PATH / "best_cnn_model.keras",
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
    ]
    
    # Data augmentation for training
    data_augmentation = keras.Sequential([
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.1),
        keras.layers.RandomTranslation(0.1, 0.1),
    ])
    
    # Create augmented training data generator
    def augment_data(X, y, augmentation_factor=3):
        """Augment training data."""
        X_aug = [X]
        y_aug = [y]
        
        for _ in range(augmentation_factor):
            X_augmented = data_augmentation(X, training=True).numpy()
            X_aug.append(X_augmented)
            y_aug.append(y)
        
        return np.concatenate(X_aug), np.concatenate(y_aug)
    
    print("\nAugmenting training data...")
    X_train_aug, y_train_aug = augment_data(X_train, y_train, augmentation_factor=3)
    print(f"Augmented training samples: {len(X_train_aug)}")
    
    # Shuffle augmented data
    shuffle_idx = np.random.permutation(len(X_train_aug))
    X_train_aug = X_train_aug[shuffle_idx]
    y_train_aug = y_train_aug[shuffle_idx]
    
    # Train
    print(f"\n{'='*50}")
    print("Training CNN Model...")
    print(f"{'='*50}")
    
    history = model.fit(
        X_train_aug, y_train_aug,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # Load best model
    model = keras.models.load_model(OUTPUT_PATH / "best_cnn_model.keras")
    
    # Evaluate
    print(f"\n{'='*50}")
    print("Evaluation Results")
    print(f"{'='*50}")
    
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
            print(f"  {class_name}: {class_acc*100:.1f}% ({mask.sum()} samples)")
    
    # Confusion matrix summary
    from sklearn.metrics import classification_report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, 
                               target_names=label_encoder.classes_))
    
    # Save model and exports
    print(f"\n{'='*50}")
    print("Saving Model...")
    print(f"{'='*50}")
    
    # Save Keras model
    model.save(OUTPUT_PATH / "asl_cnn_model.keras")
    print(f"Saved: {OUTPUT_PATH / 'asl_cnn_model.keras'}")
    
    # Save label encoder
    with open(OUTPUT_PATH / "cnn_label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
    print(f"Saved: {OUTPUT_PATH / 'cnn_label_encoder.pkl'}")
    
    # Export for TensorFlow.js
    export_for_tensorflowjs(model, label_encoder)
    
    # Also export compact weights for custom JS inference
    export_weights_json(model, label_encoder)
    
    print(f"\n✅ Training complete! Test Accuracy: {test_acc*100:.2f}%")
    
    return model, label_encoder, history

def export_for_tensorflowjs(model, label_encoder):
    """Export model for TensorFlow.js."""
    try:
        import tensorflowjs as tfjs
        tfjs_path = OUTPUT_PATH / "tfjs_model"
        tfjs.converters.save_keras_model(model, str(tfjs_path))
        print(f"Saved TensorFlow.js model: {tfjs_path}")
    except ImportError:
        print("Note: tensorflowjs not installed. Skipping TF.js export.")
        print("Install with: pip install tensorflowjs")

def export_weights_json(model, label_encoder):
    """Export model configuration and info for JavaScript."""
    config = {
        "model_type": "cnn",
        "input_shape": [IMG_SIZE, IMG_SIZE, 1],
        "num_classes": len(label_encoder.classes_),
        "classes": label_encoder.classes_.tolist(),
        "preprocessing": {
            "resize": [IMG_SIZE, IMG_SIZE],
            "grayscale": True,
            "normalize": True
        }
    }
    
    with open(OUTPUT_PATH / "cnn_model_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved: {OUTPUT_PATH / 'cnn_model_config.json'}")

if __name__ == "__main__":
    # Import keras here to handle TF initialization
    import tensorflow as tf
    from tensorflow import keras
    
    # Set memory growth to avoid OOM
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    train_model()
