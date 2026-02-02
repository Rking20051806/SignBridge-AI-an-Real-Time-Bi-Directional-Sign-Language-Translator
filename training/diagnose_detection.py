"""
Diagnostic script to analyze why MediaPipe fails to detect hands in certain letter images.
Tests D, E, G, I, P, V folders and provides detailed feedback.
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
from collections import defaultdict

# Configuration
DATASET_PATH = Path(__file__).parent.parent / "aslhandsignwithlandmark" / "own_dataset"
PROBLEM_LETTERS = ["D", "E", "G", "I", "P", "V"]
MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"

def analyze_image_quality(image_path):
    """Analyze image quality metrics."""
    img = cv2.imread(str(image_path))
    if img is None:
        return {"error": "Cannot read image"}
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate metrics
    metrics = {
        "resolution": f"{img.shape[1]}x{img.shape[0]}",
        "brightness": np.mean(gray),
        "contrast": np.std(gray),
        "blur_score": cv2.Laplacian(gray, cv2.CV_64F).var()  # Higher = sharper
    }
    
    # Quality assessment
    issues = []
    if img.shape[0] < 200 or img.shape[1] < 200:
        issues.append("Resolution too low")
    if metrics["brightness"] < 50:
        issues.append("Too dark")
    elif metrics["brightness"] > 200:
        issues.append("Too bright/overexposed")
    if metrics["contrast"] < 30:
        issues.append("Low contrast")
    if metrics["blur_score"] < 100:
        issues.append("Blurry image")
    
    metrics["issues"] = issues
    return metrics

def test_detection(image_path, detector):
    """Test hand detection on an image with multiple approaches."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None, "Cannot read image"
    
    results_info = []
    
    # Approach 1: Original image
    try:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)
        if result.hand_landmarks:
            return True, "Detected with original image"
        results_info.append("Original: No detection")
    except Exception as e:
        results_info.append(f"Original: Error - {str(e)[:30]}")
    
    # Approach 2: Resized
    try:
        resized = cv2.resize(img, (640, 640))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)
        if result.hand_landmarks:
            return True, "Detected after resize to 640x640"
        results_info.append("Resized: No detection")
    except Exception as e:
        results_info.append(f"Resized: Error - {str(e)[:30]}")
    
    # Approach 3: Brightness/contrast adjusted
    try:
        adjusted = cv2.convertScaleAbs(img, alpha=1.3, beta=30)
        rgb = cv2.cvtColor(adjusted, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)
        if result.hand_landmarks:
            return True, "Detected after brightness adjustment"
        results_info.append("Adjusted: No detection")
    except Exception as e:
        results_info.append(f"Adjusted: Error - {str(e)[:30]}")
    
    # Approach 4: Histogram equalization
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eq = cv2.equalizeHist(gray)
        eq_rgb = cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=eq_rgb)
        result = detector.detect(mp_image)
        if result.hand_landmarks:
            return True, "Detected after histogram equalization"
        results_info.append("HistEq: No detection")
    except Exception as e:
        results_info.append(f"HistEq: Error - {str(e)[:30]}")
    
    return False, " | ".join(results_info)

def main():
    print("=" * 60)
    print("MediaPipe Hand Detection Diagnostic Tool")
    print("=" * 60)
    
    # Check if model exists
    if not MODEL_PATH.exists():
        print(f"\n❌ Model not found at {MODEL_PATH}")
        print("Please run train_asl_model.py first to download the model.")
        return
    
    # Create detector with low thresholds
    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.1,  # Very low threshold
        min_tracking_confidence=0.1
    )
    
    with vision.HandLandmarker.create_from_options(options) as detector:
        
        for letter in PROBLEM_LETTERS:
            print(f"\n{'=' * 60}")
            print(f"Analyzing Letter: {letter}")
            print("=" * 60)
            
            folder = DATASET_PATH / letter
            if not folder.exists():
                print(f"  ❌ Folder not found: {folder}")
                continue
            
            images = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
            print(f"  Total images: {len(images)}")
            
            if len(images) == 0:
                print(f"  ❌ No images found")
                continue
            
            # Test sample of images
            sample_size = min(20, len(images))
            sample_images = images[:sample_size]
            
            detected_count = 0
            quality_issues = defaultdict(int)
            detection_methods = defaultdict(int)
            
            print(f"\n  Testing {sample_size} sample images...")
            
            for img_path in sample_images:
                # Quality analysis
                quality = analyze_image_quality(img_path)
                for issue in quality.get("issues", []):
                    quality_issues[issue] += 1
                
                # Detection test
                detected, method = test_detection(img_path, detector)
                if detected:
                    detected_count += 1
                    detection_methods[method] += 1
            
            # Results
            detection_rate = (detected_count / sample_size) * 100
            print(f"\n  📊 Detection Results:")
            print(f"     Detected: {detected_count}/{sample_size} ({detection_rate:.1f}%)")
            
            if detection_methods:
                print(f"\n  ✅ Successful detection methods:")
                for method, count in detection_methods.items():
                    print(f"     - {method}: {count} images")
            
            if quality_issues:
                print(f"\n  ⚠️ Image quality issues found:")
                for issue, count in sorted(quality_issues.items(), key=lambda x: -x[1]):
                    print(f"     - {issue}: {count}/{sample_size} images")
            
            # Detailed look at first failed image
            if detected_count < sample_size:
                print(f"\n  🔍 Sample failed image analysis:")
                for img_path in sample_images:
                    detected, _ = test_detection(img_path, detector)
                    if not detected:
                        quality = analyze_image_quality(img_path)
                        print(f"     File: {img_path.name}")
                        print(f"     Resolution: {quality.get('resolution', 'N/A')}")
                        print(f"     Brightness: {quality.get('brightness', 0):.1f}")
                        print(f"     Contrast: {quality.get('contrast', 0):.1f}")
                        print(f"     Sharpness: {quality.get('blur_score', 0):.1f}")
                        if quality.get('issues'):
                            print(f"     Issues: {', '.join(quality['issues'])}")
                        break

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    print("""
1. If images are blurry: Recapture with steady hand/camera
2. If too dark/bright: Improve lighting conditions
3. If low contrast: Use plain background (white/black)
4. If low resolution: Capture at higher resolution
5. Hand should be clearly visible and not cut off
6. Avoid complex backgrounds that blend with skin tone
""")

if __name__ == "__main__":
    main()
