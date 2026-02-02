"""
Image Capture Script for ASL Hand Signs
Captures ONLY the hand region with colored landmarks for better training.
Press SPACE to capture, Q to quit, N for next letter.

Finger Colors:
🔴 Thumb: Red
🟢 Index: Green
🔵 Middle: Blue
🟡 Ring: Yellow
🟣 Pinky: Magenta
🔷 Palm/Wrist: Cyan
"""

import cv2
import os
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path

# Configuration
DATASET_PATH = Path(__file__).parent / "landmark_dataset"  # New folder for landmark images
MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"
LETTERS_TO_CAPTURE = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # A-Z
IMAGES_PER_LETTER = 1  # Only capture 1, will augment to 100
AUGMENT_TO = 100  # Multiply each image to this many

# Finger colors (BGR format for OpenCV)
FINGER_COLORS = {
    'thumb': (68, 68, 255),     # Red (BGR)
    'index': (68, 255, 68),     # Green
    'middle': (255, 68, 68),    # Blue
    'ring': (68, 255, 255),     # Yellow
    'pinky': (255, 68, 255),    # Magenta
    'palm': (255, 255, 68)      # Cyan
}

# Connections with finger assignments
HAND_CONNECTIONS_COLORED = [
    # Thumb (Red)
    (0, 1, 'thumb'), (1, 2, 'thumb'), (2, 3, 'thumb'), (3, 4, 'thumb'),
    # Index (Green)
    (0, 5, 'palm'), (5, 6, 'index'), (6, 7, 'index'), (7, 8, 'index'),
    # Middle (Blue)
    (5, 9, 'palm'), (9, 10, 'middle'), (10, 11, 'middle'), (11, 12, 'middle'),
    # Ring (Yellow)
    (9, 13, 'palm'), (13, 14, 'ring'), (14, 15, 'ring'), (15, 16, 'ring'),
    # Pinky (Magenta)
    (13, 17, 'palm'), (0, 17, 'palm'), (17, 18, 'pinky'), (18, 19, 'pinky'), (19, 20, 'pinky')
]

# Landmark to finger mapping
LANDMARK_FINGERS = {
    0: 'palm',  # Wrist
    1: 'thumb', 2: 'thumb', 3: 'thumb', 4: 'thumb',
    5: 'palm', 6: 'index', 7: 'index', 8: 'index',
    9: 'palm', 10: 'middle', 11: 'middle', 12: 'middle',
    13: 'palm', 14: 'ring', 15: 'ring', 16: 'ring',
    17: 'palm', 18: 'pinky', 19: 'pinky', 20: 'pinky'
}

# Fingertip labels
FINGERTIP_LABELS = {4: 'T', 8: 'I', 12: 'M', 16: 'R', 20: 'P'}


def get_hand_bounding_box(landmarks, width, height, padding=80):
    """Get bounding box around hand with padding."""
    xs = [lm.x * width for lm in landmarks]
    ys = [lm.y * height for lm in landmarks]
    
    min_x = max(0, int(min(xs) - padding))
    max_x = min(width, int(max(xs) + padding))
    min_y = max(0, int(min(ys) - padding))
    max_y = min(height, int(max(ys) + padding))
    
    return min_x, min_y, max_x, max_y


def draw_colored_landmarks(frame, landmarks, width, height):
    """Draw hand landmarks with different colors for each finger."""
    # Draw connections first (behind landmarks)
    for start_idx, end_idx, finger in HAND_CONNECTIONS_COLORED:
        start = landmarks[start_idx]
        end = landmarks[end_idx]
        
        start_point = (int(start.x * width), int(start.y * height))
        end_point = (int(end.x * width), int(end.y * height))
        color = FINGER_COLORS[finger]
        
        cv2.line(frame, start_point, end_point, color, 4)
    
    # Draw landmarks on top
    for idx, lm in enumerate(landmarks):
        x = int(lm.x * width)
        y = int(lm.y * height)
        finger = LANDMARK_FINGERS.get(idx, 'palm')
        color = FINGER_COLORS[finger]
        
        # White outer ring
        cv2.circle(frame, (x, y), 10, (255, 255, 255), -1)
        # Colored inner dot
        cv2.circle(frame, (x, y), 7, color, -1)
        
        # Add fingertip labels
        if idx in FINGERTIP_LABELS:
            cv2.putText(frame, FINGERTIP_LABELS[idx], (x - 5, y + 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)


def extract_hand_region(frame, landmarks, width, height, output_size=224):
    """Extract and resize hand region WITHOUT landmarks (clean image for training)."""
    # Get bounding box
    min_x, min_y, max_x, max_y = get_hand_bounding_box(landmarks, width, height)
    
    # Crop CLEAN frame (no landmarks drawn) for saving
    hand_crop = frame[min_y:max_y, min_x:max_x].copy()
    
    # Make it square (pad if needed)
    h, w = hand_crop.shape[:2]
    max_dim = max(h, w)
    
    # Create square black canvas
    square = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
    
    # Center the hand crop
    y_offset = (max_dim - h) // 2
    x_offset = (max_dim - w) // 2
    square[y_offset:y_offset+h, x_offset:x_offset+w] = hand_crop
    
    # Resize to output size
    resized = cv2.resize(square, (output_size, output_size), interpolation=cv2.INTER_AREA)
    
    return resized, (min_x, min_y, max_x, max_y)


def extract_hand_region_with_landmarks(frame, landmarks, width, height, output_size=224):
    """Extract hand region WITH landmarks drawn (for preview only)."""
    min_x, min_y, max_x, max_y = get_hand_bounding_box(landmarks, width, height)
    
    # Create a copy with landmarks for preview
    hand_frame = frame.copy()
    draw_colored_landmarks(hand_frame, landmarks, width, height)
    
    hand_crop = hand_frame[min_y:max_y, min_x:max_x]
    
    h, w = hand_crop.shape[:2]
    max_dim = max(h, w)
    square = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
    y_offset = (max_dim - h) // 2
    x_offset = (max_dim - w) // 2
    square[y_offset:y_offset+h, x_offset:x_offset+w] = hand_crop
    
    resized = cv2.resize(square, (output_size, output_size), interpolation=cv2.INTER_AREA)
    return resized


def draw_color_legend(frame, x, y):
    """Draw finger color legend."""
    fingers = [
        ('Thumb', 'thumb'),
        ('Index', 'index'),
        ('Middle', 'middle'),
        ('Ring', 'ring'),
        ('Pinky', 'pinky'),
        ('Palm', 'palm')
    ]
    
    for i, (name, key) in enumerate(fingers):
        color = FINGER_COLORS[key]
        cy = y + i * 25
        cv2.circle(frame, (x, cy), 8, color, -1)
        cv2.putText(frame, name, (x + 15, cy + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def create_detector():
    """Create MediaPipe hand landmarker."""
    if not MODEL_PATH.exists():
        print(f"❌ Model not found at {MODEL_PATH}")
        print("Please run train_asl_model.py first to download the model.")
        return None
    
    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return vision.HandLandmarker.create_from_options(options)


def augment_image(img, count=100):
    """Create augmented versions of an image."""
    augmented = []
    h, w = img.shape[:2]
    
    for i in range(count):
        aug = img.copy()
        
        # Random brightness adjustment (-30 to +30)
        brightness = np.random.randint(-30, 31)
        aug = np.clip(aug.astype(np.int16) + brightness, 0, 255).astype(np.uint8)
        
        # Random rotation (-15 to +15 degrees)
        angle = np.random.uniform(-15, 15)
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        # Random zoom (0.9 to 1.1)
        zoom = np.random.uniform(0.9, 1.1)
        new_h, new_w = int(h * zoom), int(w * zoom)
        aug = cv2.resize(aug, (new_w, new_h))
        
        # Center crop back to original size
        if zoom > 1:
            start_y = (new_h - h) // 2
            start_x = (new_w - w) // 2
            aug = aug[start_y:start_y+h, start_x:start_x+w]
        else:
            pad_y = (h - new_h) // 2
            pad_x = (w - new_w) // 2
            padded = np.zeros((h, w, 3), dtype=np.uint8)
            padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = aug
            aug = padded
        
        # Random horizontal shift (-10 to +10 pixels)
        shift_x = np.random.randint(-10, 11)
        shift_y = np.random.randint(-10, 11)
        M_shift = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        aug = cv2.warpAffine(aug, M_shift, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        # Random contrast adjustment (0.8 to 1.2)
        contrast = np.random.uniform(0.8, 1.2)
        aug = np.clip((aug.astype(np.float32) - 128) * contrast + 128, 0, 255).astype(np.uint8)
        
        augmented.append(aug)
    
    return augmented


def main():
    print("=" * 60)
    print("ASL Image Capture Tool (Colored Landmarks)")
    print("1 Capture → 100 Augmented Images")
    print("=" * 60)
    print("=" * 60)
    print("\nFinger Colors:")
    print("  🔴 Thumb: Red")
    print("  🟢 Index: Green")
    print("  🔵 Middle: Blue")
    print("  🟡 Ring: Yellow")
    print("  🟣 Pinky: Magenta")
    print("  🔷 Palm/Wrist: Cyan")
    print(f"\nLetters to capture: {', '.join(LETTERS_TO_CAPTURE)}")
    print(f"Target images per letter: {IMAGES_PER_LETTER}")
    print("\nControls:")
    print("  SPACE - Capture hand region with landmarks")
    print("  N     - Next letter")
    print("  P     - Previous letter")
    print("  Q     - Quit")
    print("-" * 60)
    
    detector = create_detector()
    if detector is None:
        return
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    current_letter_idx = 0
    captured_counts = {letter: 0 for letter in LETTERS_TO_CAPTURE}
    
    # Count existing images
    print("\nExisting images:")
    for letter in LETTERS_TO_CAPTURE:
        folder = DATASET_PATH / letter
        if folder.exists():
            existing = len(list(folder.glob("*.jpg"))) + len(list(folder.glob("*.png")))
            if existing > 0:
                print(f"  {letter}: {existing} images")
    
    print("\nReady! Show your hand to the camera...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            
            # Keep a clean copy before drawing landmarks
            clean_frame = frame.copy()
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            result = detector.detect(mp_image)
            hand_detected = False
            hand_preview = None
            hand_clean = None
            bbox = None
            
            if result.hand_landmarks and len(result.hand_landmarks) > 0:
                hand_detected = True
                landmarks = result.hand_landmarks[0]
                
                # Get CLEAN hand region for saving (no landmarks)
                hand_clean, bbox = extract_hand_region(clean_frame, landmarks, width, height)
                
                # Get preview with landmarks for display
                hand_preview = extract_hand_region_with_landmarks(clean_frame, landmarks, width, height)
                
                # Draw colored landmarks on display frame
                draw_colored_landmarks(frame, landmarks, width, height)
                
                # Draw bounding box
                min_x, min_y, max_x, max_y = bbox
                cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 3)
                
                # Corner brackets
                bracket_len = 30
                # Top-left
                cv2.line(frame, (min_x, min_y), (min_x + bracket_len, min_y), (0, 255, 0), 5)
                cv2.line(frame, (min_x, min_y), (min_x, min_y + bracket_len), (0, 255, 0), 5)
                # Top-right
                cv2.line(frame, (max_x, min_y), (max_x - bracket_len, min_y), (0, 255, 0), 5)
                cv2.line(frame, (max_x, min_y), (max_x, min_y + bracket_len), (0, 255, 0), 5)
                # Bottom-left
                cv2.line(frame, (min_x, max_y), (min_x + bracket_len, max_y), (0, 255, 0), 5)
                cv2.line(frame, (min_x, max_y), (min_x, max_y - bracket_len), (0, 255, 0), 5)
                # Bottom-right
                cv2.line(frame, (max_x, max_y), (max_x - bracket_len, max_y), (0, 255, 0), 5)
                cv2.line(frame, (max_x, max_y), (max_x, max_y - bracket_len), (0, 255, 0), 5)
            
            current_letter = LETTERS_TO_CAPTURE[current_letter_idx]
            
            # UI Background
            cv2.rectangle(frame, (0, 0), (width, 90), (30, 30, 30), -1)
            
            # Letter display (big)
            cv2.putText(frame, current_letter, (30, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 4)
            
            # Capture count
            count_text = f"Captured: {captured_counts[current_letter]}/{IMAGES_PER_LETTER}"
            cv2.putText(frame, count_text, (150, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
            
            # Progress
            progress_text = f"Letter {current_letter_idx + 1}/{len(LETTERS_TO_CAPTURE)}"
            cv2.putText(frame, progress_text, (400, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
            
            # Instructions
            cv2.putText(frame, "SPACE: Capture | N: Next | P: Prev | Q: Quit", 
                       (600, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)
            
            # Color legend
            draw_color_legend(frame, 20, 110)
            
            # Status at bottom
            if hand_detected:
                cv2.rectangle(frame, (0, height - 50), (width, height), (0, 100, 0), -1)
                cv2.putText(frame, "HAND DETECTED - Press SPACE to capture", 
                           (20, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Show preview in corner
                if hand_preview is not None:
                    preview_size = 200
                    preview_resized = cv2.resize(hand_preview, (preview_size, preview_size))
                    
                    # Position in bottom-right corner
                    px, py = width - preview_size - 20, height - preview_size - 70
                    
                    # White border
                    cv2.rectangle(frame, (px - 3, py - 3), 
                                 (px + preview_size + 3, py + preview_size + 3), 
                                 (255, 255, 255), 2)
                    
                    frame[py:py + preview_size, px:px + preview_size] = preview_resized
                    
                    cv2.putText(frame, "PREVIEW (What will be saved)", 
                               (px, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                cv2.rectangle(frame, (0, height - 50), (width, height), (0, 0, 100), -1)
                cv2.putText(frame, "Show your hand clearly in the frame", 
                           (20, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            cv2.imshow("ASL Capture - Colored Landmarks", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            
            elif key == ord('n'):
                current_letter_idx = (current_letter_idx + 1) % len(LETTERS_TO_CAPTURE)
                print(f"\n→ Letter: {LETTERS_TO_CAPTURE[current_letter_idx]}")
            
            elif key == ord('p'):
                current_letter_idx = (current_letter_idx - 1) % len(LETTERS_TO_CAPTURE)
                print(f"\n→ Letter: {LETTERS_TO_CAPTURE[current_letter_idx]}")
            
            elif key == ord(' '):
                if hand_detected and hand_clean is not None:
                    save_folder = DATASET_PATH / current_letter
                    save_folder.mkdir(parents=True, exist_ok=True)
                    
                    timestamp = int(time.time() * 1000)
                    
                    # Save CLEAN original image (no landmarks - for training)
                    original_filename = f"hand_{current_letter}_{timestamp}_original.jpg"
                    cv2.imwrite(str(save_folder / original_filename), hand_clean)
                    
                    # Generate augmented images from clean image
                    print(f"  ⏳ Generating {AUGMENT_TO} augmented images (clean, no landmarks)...")
                    augmented_images = augment_image(hand_clean, AUGMENT_TO)
                    
                    for i, aug_img in enumerate(augmented_images):
                        aug_filename = f"hand_{current_letter}_{timestamp}_aug_{i:03d}.jpg"
                        cv2.imwrite(str(save_folder / aug_filename), aug_img)
                    
                    captured_counts[current_letter] += 1
                    print(f"  ✓ Saved: 1 original + {AUGMENT_TO} augmented = {AUGMENT_TO + 1} CLEAN images for '{current_letter}'")
                    
                    # Flash effect
                    flash = frame.copy()
                    cv2.rectangle(flash, (0, 0), (width, height), (0, 255, 0), -1)
                    cv2.putText(flash, f"{AUGMENT_TO + 1} IMAGES SAVED!", (width//2 - 200, height//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
                    cv2.imshow("ASL Capture - Colored Landmarks", flash)
                    cv2.waitKey(500)
                    
                    # Auto-advance
                    if captured_counts[current_letter] >= IMAGES_PER_LETTER:
                        print(f"\n✅ Completed {current_letter}!")
                        if current_letter_idx < len(LETTERS_TO_CAPTURE) - 1:
                            current_letter_idx += 1
                            print(f"→ Next: {LETTERS_TO_CAPTURE[current_letter_idx]}")
                        else:
                            print("\n🎉 ALL LETTERS COMPLETED!")
                else:
                    print("  ⚠️ No hand detected!")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        
        print("\n" + "=" * 60)
        print("CAPTURE SUMMARY")
        print("=" * 60)
        total = sum(captured_counts.values())
        print(f"Letters captured: {total}")
        print(f"Total images: {total * (AUGMENT_TO + 1)}")
        for letter, count in captured_counts.items():
            if count > 0:
                print(f"  ✅ {letter}: {count * (AUGMENT_TO + 1)} images")
        
        if total > 0:
            print(f"\n💡 Run 'python train_asl_model.py' to retrain!")


if __name__ == "__main__":
    main()
