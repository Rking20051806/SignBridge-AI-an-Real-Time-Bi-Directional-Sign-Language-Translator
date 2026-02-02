import React, { useRef, useState, useEffect, useCallback } from 'react';
import { Camera, Download, Trash2, Play, Square, FolderOpen, Save, ChevronLeft, ChevronRight, Loader2, CheckCircle2, AlertCircle, Laptop, Smartphone, Link, Wifi, Brain, Zap } from 'lucide-react';
import { HandLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import { trainModel, getCustomModels, downloadModelAsJson, type TrainingProgress, type TrainedModel } from '../services/modelTrainer';

type CameraSource = 'webcam' | 'ip-camera';

// Finger colors for landmarks
const FINGER_COLORS: Record<string, string> = {
  thumb: "#FF4444",
  index: "#44FF44",
  middle: "#4444FF",
  ring: "#FFFF44",
  pinky: "#FF44FF",
  palm: "#44FFFF"
};

const HAND_CONNECTIONS = [
  { start: 0, end: 1, finger: 'thumb' }, { start: 1, end: 2, finger: 'thumb' },
  { start: 2, end: 3, finger: 'thumb' }, { start: 3, end: 4, finger: 'thumb' },
  { start: 0, end: 5, finger: 'palm' }, { start: 5, end: 6, finger: 'index' },
  { start: 6, end: 7, finger: 'index' }, { start: 7, end: 8, finger: 'index' },
  { start: 5, end: 9, finger: 'palm' }, { start: 9, end: 10, finger: 'middle' },
  { start: 10, end: 11, finger: 'middle' }, { start: 11, end: 12, finger: 'middle' },
  { start: 9, end: 13, finger: 'palm' }, { start: 13, end: 14, finger: 'ring' },
  { start: 14, end: 15, finger: 'ring' }, { start: 15, end: 16, finger: 'ring' },
  { start: 13, end: 17, finger: 'palm' }, { start: 0, end: 17, finger: 'palm' },
  { start: 17, end: 18, finger: 'pinky' }, { start: 18, end: 19, finger: 'pinky' },
  { start: 19, end: 20, finger: 'pinky' }
];

const LANDMARK_FINGERS: Record<number, string> = {
  0: 'palm', 1: 'thumb', 2: 'thumb', 3: 'thumb', 4: 'thumb',
  5: 'palm', 6: 'index', 7: 'index', 8: 'index',
  9: 'palm', 10: 'middle', 11: 'middle', 12: 'middle',
  13: 'palm', 14: 'ring', 15: 'ring', 16: 'ring',
  17: 'palm', 18: 'pinky', 19: 'pinky', 20: 'pinky'
};

interface CapturedImage {
  id: string;
  letter: string; // Now can be any sign name
  imageData: string; // Base64 with landmarks
  landmarks: number[][]; // 21 landmarks x 3 coordinates
  timestamp: number;
}

const DatasetCapture: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const imgRef = useRef<HTMLImageElement>(null); // For MJPEG streams
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const animationRef = useRef<number>(0);
  
  // Camera source selection
  const [cameraSource, setCameraSource] = useState<CameraSource>('webcam');
  const [ipCameraUrl, setIpCameraUrl] = useState('http://172.16.22.11:4747/video');
  const [showCameraSettings, setShowCameraSettings] = useState(false);
  const [useImageMode, setUseImageMode] = useState(false); // For MJPEG streams
  
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [isStreaming, setIsStreaming] = useState(false);
  
  // Custom signs - start empty, user adds their own
  const [customSigns, setCustomSigns] = useState<string[]>([]);
  const [currentSign, setCurrentSign] = useState('');
  const [newSignInput, setNewSignInput] = useState('');
  
  const [capturedImages, setCapturedImages] = useState<CapturedImage[]>([]);
  const [currentLandmarks, setCurrentLandmarks] = useState<any[] | null>(null);
  const [autoCapture, setAutoCapture] = useState(false);
  const [captureCount, setCaptureCount] = useState(0);
  const [targetCount, setTargetCount] = useState(50);
  const [message, setMessage] = useState<string | null>(null);
  const [selectedHandIndex, setSelectedHandIndex] = useState(0); // Which hand to capture (0 = first detected, 1 = second)
  const [handCount, setHandCount] = useState(0); // Just track count, not full landmarks array
  const currentBoundingBoxRef = useRef<{x: number, y: number, width: number, height: number} | null>(null);

  // Training state
  const [modelName, setModelName] = useState('my_custom_model');
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState<TrainingProgress | null>(null);
  const [trainedModels, setTrainedModels] = useState<{ name: string; trainedAt: string; accuracy: number }[]>([]);
  const [showTrainingPanel, setShowTrainingPanel] = useState(false);

  // Load trained models on mount
  useEffect(() => {
    setTrainedModels(getCustomModels());
  }, []);

  // Load MediaPipe HandLandmarker
  useEffect(() => {
    const loadModel = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
        );
        
        handLandmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numHands: 2, // Detect both hands - user can use either hand for same sign
          minHandDetectionConfidence: 0.5,
          minHandPresenceConfidence: 0.5,
          minTrackingConfidence: 0.5
        });
        
        setIsModelLoading(false);
        console.log("✅ HandLandmarker loaded for capture");
      } catch (err) {
        console.error("Failed to load model:", err);
        setMessage("Failed to load hand detection model");
      }
    };
    loadModel();
    
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
    };
  }, []);

  // Start camera
  const startCamera = async () => {
    if (!handLandmarkerRef.current) return;
    
    try {
      if (cameraSource === 'webcam') {
        // Use laptop/desktop webcam with video element
        if (!videoRef.current) return;
        setUseImageMode(false);
        
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }
        });
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        setIsStreaming(true);
        setMessage('Webcam connected ✓');
        detectLoopVideo();
      } else {
        // Use IP camera with img element (MJPEG stream)
        if (!imgRef.current) return;
        if (!ipCameraUrl) {
          setMessage("Please enter IP camera URL");
          return;
        }
        
        setUseImageMode(true);
        setMessage("Connecting to IP camera...");
        
        // For MJPEG streams, use an img element
        imgRef.current.src = ipCameraUrl;
        
        // Wait for image to load
        await new Promise<void>((resolve, reject) => {
          const img = imgRef.current!;
          const timeout = setTimeout(() => {
            reject(new Error("Connection timeout - check DroidCam is running"));
          }, 10000);
          
          img.onload = () => {
            clearTimeout(timeout);
            resolve();
          };
          
          img.onerror = () => {
            clearTimeout(timeout);
            reject(new Error("Cannot load stream - check URL"));
          };
        });
        
        setIsStreaming(true);
        setMessage('IP camera connected ✓');
        detectLoopImage();
      }
    } catch (err: any) {
      console.error('Camera error:', err);
      setMessage(`Error: ${err.message}`);
    }
  };
  
  // Detection loop for video element (webcam)
  const detectLoopVideo = () => {
    if (!videoRef.current || !canvasRef.current || !handLandmarkerRef.current) return;
    if (videoRef.current.paused || videoRef.current.ended) return;
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const shouldMirror = true; // Always mirror webcam
    
    if (ctx && video.readyState >= 2) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      if (shouldMirror) {
        ctx.save();
        ctx.scale(-1, 1);
        ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
        ctx.restore();
      } else {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      }
      
      const result = handLandmarkerRef.current.detectForVideo(video, performance.now());
      
      if (result.landmarks && result.landmarks.length > 0) {
        // Draw all detected hands, highlight selected one
        result.landmarks.forEach((landmarks, idx) => {
          const isSelected = idx === selectedHandIndex || result.landmarks.length === 1;
          drawLandmarks(ctx, landmarks, canvas.width, canvas.height, shouldMirror, isSelected);
        });
        
        // Use selected hand (or first if only one detected)
        const activeIdx = result.landmarks.length > selectedHandIndex ? selectedHandIndex : 0;
        
        // Only update state if values actually changed
        if (result.landmarks.length !== handCount) {
          setHandCount(result.landmarks.length);
        }
        setCurrentLandmarks(result.landmarks[activeIdx]);
      } else {
        if (currentLandmarks !== null) {
          setCurrentLandmarks(null);
          setHandCount(0);
          currentBoundingBoxRef.current = null;
        }
      }
    }
    
    animationRef.current = requestAnimationFrame(detectLoopVideo);
  };
  
  // Detection loop for img element (IP camera MJPEG)
  const detectLoopImage = () => {
    if (!imgRef.current || !canvasRef.current || !handLandmarkerRef.current) return;
    
    const img = imgRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    if (ctx && img.complete && img.naturalWidth > 0) {
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      
      // Draw the MJPEG frame
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      
      // Create an ImageData for MediaPipe
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      
      // Detect using IMAGE mode (not VIDEO mode for still images)
      try {
        const result = handLandmarkerRef.current.detect(img);
        
        if (result.landmarks && result.landmarks.length > 0) {
          // Draw all detected hands, highlight selected one
          result.landmarks.forEach((landmarks, idx) => {
            const isSelected = idx === selectedHandIndex || result.landmarks.length === 1;
            drawLandmarks(ctx, landmarks, canvas.width, canvas.height, false, isSelected);
          });
          
          const activeIdx = result.landmarks.length > selectedHandIndex ? selectedHandIndex : 0;
          if (result.landmarks.length !== handCount) {
            setHandCount(result.landmarks.length);
          }
          setCurrentLandmarks(result.landmarks[activeIdx]);
        } else {
          if (currentLandmarks !== null) {
            setCurrentLandmarks(null);
            setHandCount(0);
            currentBoundingBoxRef.current = null;
          }
        }
      } catch (e) {
        // Silently ignore detection errors on frames
      }
    }
    
    animationRef.current = requestAnimationFrame(detectLoopImage);
  };

  // Stop camera
  const stopCamera = () => {
    if (animationRef.current) cancelAnimationFrame(animationRef.current);
    
    // Stop webcam
    if (videoRef.current) {
      if (videoRef.current.srcObject) {
        (videoRef.current.srcObject as MediaStream).getTracks().forEach(t => t.stop());
        videoRef.current.srcObject = null;
      }
      videoRef.current.src = '';
      videoRef.current.pause();
    }
    
    // Stop IP camera image
    if (imgRef.current) {
      imgRef.current.src = '';
    }
    
    setIsStreaming(false);
    setCurrentLandmarks(null);
    setUseImageMode(false);
  };

  // Draw landmarks with colors and bounding box
  const drawLandmarks = (
    ctx: CanvasRenderingContext2D, 
    landmarks: any[], 
    width: number, 
    height: number,
    mirror: boolean = false,
    isSelectedHand: boolean = true // Highlight the selected hand
  ) => {
    // Calculate and draw bounding box for selected hand
    const bbox = getHandBoundingBox(landmarks, width, height);
    
    // Draw bounding box
    ctx.strokeStyle = isSelectedHand ? '#00FF00' : '#FFFF00'; // Green for selected, yellow for other
    ctx.lineWidth = isSelectedHand ? 4 : 2;
    ctx.setLineDash(isSelectedHand ? [] : [10, 5]);
    
    // Adjust box coordinates for mirroring
    let boxX = bbox.x;
    if (mirror) {
      boxX = width - bbox.x - bbox.width;
    }
    
    ctx.strokeRect(boxX, bbox.y, bbox.width, bbox.height);
    ctx.setLineDash([]);
    
    // Draw "CAPTURE AREA" label on selected hand
    if (isSelectedHand) {
      ctx.fillStyle = '#00FF00';
      ctx.font = 'bold 14px Arial';
      ctx.fillText('✓ CAPTURE ZONE', boxX + 5, bbox.y - 8);
      
      // Store bounding box for capture (using ref, not state - avoids re-renders)
      currentBoundingBoxRef.current = bbox;
    }
    
    // Draw connections
    ctx.lineWidth = isSelectedHand ? 4 : 2;
    for (const conn of HAND_CONNECTIONS) {
      const start = landmarks[conn.start];
      const end = landmarks[conn.end];
      const color = FINGER_COLORS[conn.finger];
      
      let sx = start.x * width;
      let ex = end.x * width;
      
      if (mirror) {
        sx = width - sx;
        ex = width - ex;
      }
      
      ctx.strokeStyle = isSelectedHand ? color : color + '80'; // Dimmed for non-selected hand
      ctx.beginPath();
      ctx.moveTo(sx, start.y * height);
      ctx.lineTo(ex, end.y * height);
      ctx.stroke();
    }
    
    // Draw landmarks
    landmarks.forEach((lm: any, idx: number) => {
      const finger = LANDMARK_FINGERS[idx] || 'palm';
      const color = FINGER_COLORS[finger];
      
      let x = lm.x * width;
      if (mirror) x = width - x;
      const y = lm.y * height;
      
      // White border
      ctx.fillStyle = "#FFFFFF";
      ctx.beginPath();
      ctx.arc(x, y, 8, 0, Math.PI * 2);
      ctx.fill();
      
      // Colored center
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fill();
    });
  };

  // Get hand bounding box
  const getHandBoundingBox = (landmarks: any[], width: number, height: number, padding: number = 60) => {
    const xs = landmarks.map(l => l.x * width);
    const ys = landmarks.map(l => l.y * height);
    
    const minX = Math.max(0, Math.min(...xs) - padding);
    const maxX = Math.min(width, Math.max(...xs) + padding);
    const minY = Math.max(0, Math.min(...ys) - padding);
    const maxY = Math.min(height, Math.max(...ys) + padding);
    
    // Make it square
    const size = Math.max(maxX - minX, maxY - minY);
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    
    return {
      x: Math.max(0, centerX - size / 2),
      y: Math.max(0, centerY - size / 2),
      width: size,
      height: size
    };
  };

  // Capture image with landmarks
  const captureImage = useCallback(() => {
    if (!currentSign) {
      setMessage("Please add and select a sign name first!");
      return;
    }
    
    if (!currentLandmarks || !canvasRef.current) {
      setMessage("No hand detected! Show your hand to the camera.");
      return;
    }
    
    const canvas = canvasRef.current;
    const width = canvas.width;
    const height = canvas.height;
    const shouldMirror = cameraSource === 'webcam';
    
    // Get bounding box
    const bbox = getHandBoundingBox(currentLandmarks, width, height);
    
    // Create cropped canvas with landmarks
    const cropCanvas = document.createElement('canvas');
    const cropSize = 224; // Standard size for training
    cropCanvas.width = cropSize;
    cropCanvas.height = cropSize;
    const cropCtx = cropCanvas.getContext('2d')!;
    
    // Draw cropped region
    cropCtx.drawImage(
      canvas,
      bbox.x, bbox.y, bbox.width, bbox.height,
      0, 0, cropSize, cropSize
    );
    
    // Draw landmarks on cropped image
    const scaleX = cropSize / bbox.width;
    const scaleY = cropSize / bbox.height;
    
    // Draw connections
    cropCtx.lineWidth = 3;
    for (const conn of HAND_CONNECTIONS) {
      const start = currentLandmarks[conn.start];
      const end = currentLandmarks[conn.end];
      const color = FINGER_COLORS[conn.finger];
      
      // Calculate positions, mirror only for webcam
      const sx = shouldMirror 
        ? (width - start.x * width - bbox.x) * scaleX 
        : (start.x * width - bbox.x) * scaleX;
      const sy = (start.y * height - bbox.y) * scaleY;
      const ex = shouldMirror 
        ? (width - end.x * width - bbox.x) * scaleX 
        : (end.x * width - bbox.x) * scaleX;
      const ey = (end.y * height - bbox.y) * scaleY;
      
      cropCtx.strokeStyle = color;
      cropCtx.beginPath();
      cropCtx.moveTo(sx, sy);
      cropCtx.lineTo(ex, ey);
      cropCtx.stroke();
    }
    
    // Draw landmark dots
    currentLandmarks.forEach((lm: any, idx: number) => {
      const finger = LANDMARK_FINGERS[idx] || 'palm';
      const color = FINGER_COLORS[finger];
      
      const x = shouldMirror 
        ? (width - lm.x * width - bbox.x) * scaleX 
        : (lm.x * width - bbox.x) * scaleX;
      const y = (lm.y * height - bbox.y) * scaleY;
      
      cropCtx.fillStyle = "#FFFFFF";
      cropCtx.beginPath();
      cropCtx.arc(x, y, 6, 0, Math.PI * 2);
      cropCtx.fill();
      
      cropCtx.fillStyle = color;
      cropCtx.beginPath();
      cropCtx.arc(x, y, 4, 0, Math.PI * 2);
      cropCtx.fill();
    });
    
    // Convert to base64
    const imageData = cropCanvas.toDataURL('image/png');
    
    // Extract landmark coordinates (normalized)
    const landmarkCoords = currentLandmarks.map((lm: any) => [lm.x, lm.y, lm.z]);
    
    // Save captured image
    const newImage: CapturedImage = {
      id: `${currentSign}_${Date.now()}`,
      letter: currentSign,
      imageData,
      landmarks: landmarkCoords,
      timestamp: Date.now()
    };
    
    setCapturedImages(prev => [...prev, newImage]);
    setCaptureCount(prev => prev + 1);
    setMessage(`Captured "${currentSign}" (${captureCount + 1}/${targetCount})`);
    
    // Flash effect
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      if (ctx) {
        ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
      }
    }
  }, [currentLandmarks, currentSign, captureCount, targetCount, cameraSource]);

  // Auto capture effect
  useEffect(() => {
    if (!autoCapture || !isStreaming || captureCount >= targetCount || !currentSign) {
      return;
    }
    
    const interval = setInterval(() => {
      if (currentLandmarks) {
        captureImage();
      }
    }, 500); // Capture every 500ms
    
    return () => clearInterval(interval);
  }, [autoCapture, isStreaming, currentLandmarks, captureCount, targetCount, captureImage, currentSign]);

  // Add new sign
  const addNewSign = () => {
    const signName = newSignInput.trim().toUpperCase();
    if (!signName) {
      setMessage("Please enter a sign name!");
      return;
    }
    if (customSigns.includes(signName)) {
      setMessage(`Sign "${signName}" already exists!`);
      return;
    }
    setCustomSigns(prev => [...prev, signName]);
    setCurrentSign(signName);
    setCaptureCount(0);
    setNewSignInput('');
    setMessage(`Added new sign: "${signName}"`);
  };

  // Navigate signs
  const prevSign = () => {
    const idx = customSigns.indexOf(currentSign);
    if (idx > 0) {
      setCurrentSign(customSigns[idx - 1]);
      setCaptureCount(0);
    }
  };

  const nextSign = () => {
    const idx = customSigns.indexOf(currentSign);
    if (idx < customSigns.length - 1) {
      setCurrentSign(customSigns[idx + 1]);
      setCaptureCount(0);
    }
  };

  // Remove a sign
  const removeSign = (signName: string) => {
    setCustomSigns(prev => prev.filter(s => s !== signName));
    setCapturedImages(prev => prev.filter(img => img.letter !== signName));
    if (currentSign === signName) {
      setCurrentSign(customSigns[0] || '');
    }
    setMessage(`Removed sign "${signName}"`);
  };

  // Download dataset as ZIP
  const downloadDataset = async () => {
    if (capturedImages.length === 0) {
      setMessage("No images captured yet!");
      return;
    }
    
    setMessage("Preparing download...");
    
    // Group by letter
    const byLetter: Record<string, CapturedImage[]> = {};
    capturedImages.forEach(img => {
      if (!byLetter[img.letter]) byLetter[img.letter] = [];
      byLetter[img.letter].push(img);
    });
    
    // Create downloadable content
    const dataStr = JSON.stringify({
      metadata: {
        totalImages: capturedImages.length,
        letters: Object.keys(byLetter),
        capturedAt: new Date().toISOString()
      },
      images: capturedImages.map(img => ({
        letter: img.letter,
        imageData: img.imageData,
        landmarks: img.landmarks
      }))
    }, null, 2);
    
    // Download JSON (can be processed by Python script)
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `asl_dataset_${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
    
    setMessage(`Downloaded ${capturedImages.length} images!`);
  };

  // Download individual images (for direct folder saving)
  const downloadImages = async () => {
    if (capturedImages.length === 0) {
      setMessage("No images captured yet!");
      return;
    }
    
    setMessage("Downloading images...");
    
    // Download each image
    for (const img of capturedImages) {
      const a = document.createElement('a');
      a.href = img.imageData;
      a.download = `${img.letter}/${img.letter}_${img.timestamp}.png`;
      // Note: Browser will typically save to Downloads folder
      // User needs to organize into folders
    }
    
    // Better: Create a organized JSON that Python can process
    const dataStr = JSON.stringify({
      images: capturedImages.map(img => ({
        letter: img.letter,
        filename: `${img.letter}_${img.timestamp}.png`,
        imageData: img.imageData,
        landmarks: img.landmarks
      }))
    });
    
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `asl_captured_dataset.json`;
    a.click();
    URL.revokeObjectURL(url);
    
    // Also save to localStorage for training
    saveDatasetToStorage();
    
    setMessage(`Downloaded dataset with ${capturedImages.length} images!`);
  };

  // Save dataset to localStorage for browser-based training
  const saveDatasetToStorage = () => {
    if (capturedImages.length === 0) return;
    
    const dataStr = JSON.stringify({
      images: capturedImages.map(img => ({
        letter: img.letter,
        landmarks: img.landmarks
      }))
    });
    
    localStorage.setItem('asl_captured_dataset', dataStr);
    console.log(`✅ Saved ${capturedImages.length} images to localStorage`);
  };

  // Train model on captured data
  const handleTrainModel = async () => {
    if (capturedImages.length === 0) {
      setMessage("No images captured! Capture some images first.");
      return;
    }
    
    if (!modelName.trim()) {
      setMessage("Please enter a model name!");
      return;
    }
    
    // Check minimum requirements
    const letterCounts = new Map<string, number>();
    capturedImages.forEach(img => {
      letterCounts.set(img.letter, (letterCounts.get(img.letter) || 0) + 1);
    });
    
    if (letterCounts.size < 2) {
      setMessage("Need at least 2 different letters to train!");
      return;
    }
    
    // Save to localStorage first
    saveDatasetToStorage();
    
    setIsTraining(true);
    setTrainingProgress(null);
    setMessage("Starting training...");
    
    try {
      const trainingData = capturedImages.map(img => ({
        letter: img.letter,
        landmarks: img.landmarks
      }));
      
      const model = await trainModel(
        trainingData,
        modelName.trim(),
        100, // epochs
        (progress) => {
          setTrainingProgress(progress);
        }
      );
      
      setMessage(`✅ Model "${model.name}" trained successfully! Accuracy: ${(model.accuracy * 100).toFixed(1)}%`);
      setTrainedModels(getCustomModels());
      
      // Optionally download the model
      downloadModelAsJson(model);
      
    } catch (error: any) {
      console.error('Training failed:', error);
      setMessage(`❌ Training failed: ${error.message}`);
    } finally {
      setIsTraining(false);
      setTrainingProgress(null);
    }
  };

  // Clear images for current sign
  const clearCurrentSign = () => {
    if (!currentSign) return;
    setCapturedImages(prev => prev.filter(img => img.letter !== currentSign));
    setCaptureCount(0);
    setMessage(`Cleared all "${currentSign}" images`);
  };

  // Get count for current sign
  const currentSignCount = currentSign ? capturedImages.filter(img => img.letter === currentSign).length : 0;

  // Get all unique signs that have captured images
  const signsWithImages = [...new Set(capturedImages.map(i => i.letter))];

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 sm:gap-6 h-full">
      {/* Main Capture Area */}
      <div className="md:col-span-2 flex flex-col gap-3 sm:gap-4">
        {/* Header */}
        <div className="bg-slate-800/50 rounded-lg sm:rounded-xl p-3 sm:p-4 border border-slate-700">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-base sm:text-xl font-bold text-white">Dataset Capture Tool</h2>
              <p className="text-xs sm:text-sm text-slate-400 hidden sm:block">Capture hand signs with landmarks for training</p>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-[10px] sm:text-xs text-slate-500">Total:</span>
              <span className="text-base sm:text-lg font-bold text-emerald-400">{capturedImages.length}</span>
            </div>
          </div>
        </div>

        {/* Video Feed */}
        <div className="relative bg-black rounded-xl overflow-hidden aspect-video border border-slate-700">
          {/* Hidden video element for webcam */}
          <video ref={videoRef} className="hidden" playsInline />
          {/* Hidden img element for MJPEG IP camera streams */}
          <img ref={imgRef} className="hidden" alt="IP Camera Feed" crossOrigin="anonymous" />
          {/* Visible canvas for rendering with landmarks */}
          <canvas ref={canvasRef} className={`w-full h-full object-contain ${cameraSource === 'webcam' ? 'scale-x-[-1]' : ''}`} />
          
          {/* Overlay when not streaming */}
          {!isStreaming && (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-900/80">
              <Camera className="w-16 h-16 text-slate-600 mb-4" />
              
              {/* Camera Source Selection */}
              <div className="mb-6 bg-slate-800/80 rounded-xl p-4 border border-slate-700 w-full max-w-md">
                <h3 className="text-sm font-bold text-slate-400 uppercase mb-3 flex items-center gap-2">
                  <Wifi className="w-4 h-4" /> Camera Source
                </h3>
                
                {/* Source Toggle */}
                <div className="flex gap-2 mb-4">
                  <button
                    onClick={() => setCameraSource('webcam')}
                    className={`flex-1 px-4 py-3 rounded-lg font-bold flex items-center justify-center gap-2 transition-all ${
                      cameraSource === 'webcam'
                        ? 'bg-blue-600 text-white'
                        : 'bg-slate-700 text-slate-400 hover:bg-slate-600'
                    }`}
                  >
                    <Laptop className="w-5 h-5" />
                    Webcam
                  </button>
                  <button
                    onClick={() => setCameraSource('ip-camera')}
                    className={`flex-1 px-4 py-3 rounded-lg font-bold flex items-center justify-center gap-2 transition-all ${
                      cameraSource === 'ip-camera'
                        ? 'bg-purple-600 text-white'
                        : 'bg-slate-700 text-slate-400 hover:bg-slate-600'
                    }`}
                  >
                    <Smartphone className="w-5 h-5" />
                    IP Camera
                  </button>
                </div>
                
                {/* IP Camera URL Input */}
                {cameraSource === 'ip-camera' && (
                  <div className="space-y-3">
                    <label className="text-xs text-slate-500 flex items-center gap-1">
                      <Link className="w-3 h-3" /> Camera URL (DroidCam, IP Webcam, etc.)
                    </label>
                    <input
                      type="text"
                      value={ipCameraUrl}
                      onChange={(e) => setIpCameraUrl(e.target.value)}
                      placeholder="http://192.168.1.100:4747/video"
                      className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded-lg text-white text-sm focus:outline-none focus:border-purple-500"
                    />
                    
                    {/* Quick URL buttons */}
                    <div className="flex flex-wrap gap-1">
                      <button
                        onClick={() => setIpCameraUrl('http://172.16.22.11:4747/video')}
                        className="px-2 py-1 bg-slate-700 hover:bg-slate-600 rounded text-[10px] text-slate-300"
                      >
                        DroidCam :4747
                      </button>
                      <button
                        onClick={() => setIpCameraUrl('http://172.16.22.11:8080/video')}
                        className="px-2 py-1 bg-slate-700 hover:bg-slate-600 rounded text-[10px] text-slate-300"
                      >
                        IP Webcam :8080
                      </button>
                      <button
                        onClick={() => setIpCameraUrl('http://172.16.22.11:4747/mjpegfeed')}
                        className="px-2 py-1 bg-slate-700 hover:bg-slate-600 rounded text-[10px] text-slate-300"
                      >
                        MJPEG Feed
                      </button>
                    </div>
                    
                    <div className="bg-slate-900/50 rounded p-2 text-[10px] text-slate-500 space-y-1">
                      <p className="text-amber-400 font-bold">⚠️ Troubleshooting:</p>
                      <p>1. Ensure phone & laptop are on same WiFi</p>
                      <p>2. DroidCam app must be running on phone</p>
                      <p>3. Try opening URL directly in Chrome first</p>
                      <p>4. Check firewall isn't blocking connection</p>
                    </div>
                  </div>
                )}
              </div>
              
              <button
                onClick={startCamera}
                disabled={isModelLoading}
                className="px-6 py-3 bg-blue-600 hover:bg-blue-500 text-white rounded-lg font-bold flex items-center gap-2 disabled:opacity-50"
              >
                {isModelLoading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Loading Model...
                  </>
                ) : (
                  <>
                    <Play className="w-5 h-5" />
                    Start {cameraSource === 'webcam' ? 'Webcam' : 'IP Camera'}
                  </>
                )}
              </button>
            </div>
          )}

          {/* Current Sign Overlay */}
          {isStreaming && (
            <div className="absolute top-4 left-4 bg-black/70 backdrop-blur px-4 py-2 rounded-lg">
              <span className="text-xs text-slate-400">Capturing:</span>
              <span className="text-2xl font-bold text-white ml-2">{currentSign || 'No sign selected'}</span>
              {currentSign && <span className="text-sm text-slate-400 ml-2">({currentSignCount}/{targetCount})</span>}
            </div>
          )}

          {/* Hand Detection Status */}
          {isStreaming && (
            <div className={`absolute top-4 right-4 px-3 py-1 rounded-full text-xs font-bold ${
              currentLandmarks ? 'bg-emerald-500/20 text-emerald-400' : 'bg-red-500/20 text-red-400'
            }`}>
              {currentLandmarks 
                ? `✓ ${handCount} Hand${handCount > 1 ? 's' : ''} Detected` 
                : '✗ No Hand'}
            </div>
          )}
          
          {/* Hand selector when multiple hands detected */}
          {isStreaming && handCount > 1 && (
            <div className="absolute top-12 right-4 bg-black/70 backdrop-blur px-3 py-2 rounded-lg">
              <span className="text-xs text-slate-400 block mb-1">Select hand:</span>
              <div className="flex gap-1">
                {[0, 1].map((idx) => (
                  <button
                    key={idx}
                    onClick={() => setSelectedHandIndex(idx)}
                    className={`px-3 py-1 rounded text-xs font-bold ${
                      selectedHandIndex === idx 
                        ? 'bg-green-600 text-white' 
                        : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                    }`}
                  >
                    Hand {idx + 1}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Controls */}
        <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
          {/* Add New Sign */}
          <div className="flex items-center gap-2 mb-4">
            <input
              type="text"
              value={newSignInput}
              onChange={(e) => setNewSignInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && addNewSign()}
              placeholder="Enter sign name (e.g., HELLO, YES, NO)"
              className="flex-1 px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white text-sm focus:outline-none focus:border-blue-500"
            />
            <button
              onClick={addNewSign}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg font-bold flex items-center gap-2"
            >
              <Camera className="w-4 h-4" />
              Add Sign
            </button>
          </div>

          <div className="flex flex-wrap items-center justify-between gap-4">
            {/* Sign Navigation */}
            <div className="flex items-center gap-2 flex-wrap">
              {customSigns.length > 0 ? (
                <>
                  <button onClick={prevSign} disabled={customSigns.indexOf(currentSign) <= 0} className="p-2 bg-slate-700 hover:bg-slate-600 disabled:opacity-50 rounded-lg" title="Previous sign">
                    <ChevronLeft className="w-5 h-5" />
                  </button>
                  <div className="flex gap-1 flex-wrap max-w-[500px]">
                    {customSigns.map(sign => (
                      <button
                        key={sign}
                        onClick={() => { setCurrentSign(sign); setCaptureCount(0); }}
                        className={`px-3 py-1 rounded text-sm font-bold transition-all ${
                          sign === currentSign
                            ? 'bg-blue-600 text-white'
                            : capturedImages.some(img => img.letter === sign)
                              ? 'bg-emerald-600/30 text-emerald-400 hover:bg-emerald-600/50'
                              : 'bg-slate-700 text-slate-400 hover:bg-slate-600'
                        }`}
                      >
                        {sign}
                      </button>
                    ))}
                  </div>
                  <button onClick={nextSign} disabled={customSigns.indexOf(currentSign) >= customSigns.length - 1} className="p-2 bg-slate-700 hover:bg-slate-600 disabled:opacity-50 rounded-lg" title="Next sign">
                    <ChevronRight className="w-5 h-5" />
                  </button>
                </>
              ) : (
                <span className="text-sm text-slate-500 italic">Add a sign name above to start capturing</span>
              )}
            </div>

            {/* Capture Controls */}
            <div className="flex items-center gap-2">
              {isStreaming && currentSign && (
                <>
                  <button
                    onClick={captureImage}
                    disabled={!currentLandmarks}
                    className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white rounded-lg font-bold flex items-center gap-2"
                  >
                    <Camera className="w-4 h-4" />
                    Capture
                  </button>
                  <button
                    onClick={() => setAutoCapture(!autoCapture)}
                    className={`px-4 py-2 rounded-lg font-bold flex items-center gap-2 ${
                      autoCapture ? 'bg-red-600 hover:bg-red-500 text-white' : 'bg-slate-700 hover:bg-slate-600 text-slate-300'
                    }`}
                  >
                    {autoCapture ? <Square className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                    {autoCapture ? 'Stop Auto' : 'Auto Capture'}
                  </button>
                  <button
                    onClick={stopCamera}
                    className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-lg font-bold"
                  >
                    Stop
                  </button>
                </>
              )}
              {isStreaming && !currentSign && (
                <span className="text-sm text-amber-400">← Add a sign name first</span>
              )}
            </div>
          </div>

          {/* Target Count */}
          <div className="flex items-center gap-4 mt-4">
            <span className="text-sm text-slate-400">Images per letter:</span>
            <input
              type="range"
              min="10"
              max="200"
              value={targetCount}
              onChange={(e) => setTargetCount(parseInt(e.target.value))}
              className="flex-1"
            />
            <span className="text-sm font-bold text-white">{targetCount}</span>
          </div>
        </div>

        {/* Message */}
        {message && (
          <div className="bg-blue-500/10 border border-blue-500/20 rounded-lg p-3 text-blue-400 text-sm flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4" />
            {message}
          </div>
        )}
      </div>

      {/* Sidebar */}
      <div className="flex flex-col gap-4">
        {/* Stats */}
        <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
          <h3 className="text-sm font-bold text-slate-400 uppercase mb-3">Dataset Stats</h3>
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-slate-900/50 rounded-lg p-3 text-center">
              <div className="text-2xl font-bold text-white">{capturedImages.length}</div>
              <div className="text-xs text-slate-500">Total Images</div>
            </div>
            <div className="bg-slate-900/50 rounded-lg p-3 text-center">
              <div className="text-2xl font-bold text-emerald-400">
                {signsWithImages.length}
              </div>
              <div className="text-xs text-slate-500">Signs</div>
            </div>
          </div>
        </div>

        {/* Per-Sign Stats */}
        <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700 flex-1 overflow-auto">
          <h3 className="text-sm font-bold text-slate-400 uppercase mb-3">Per Sign</h3>
          {customSigns.length > 0 ? (
            <div className="space-y-2">
              {customSigns.map(sign => {
                const count = capturedImages.filter(i => i.letter === sign).length;
                return (
                  <div
                    key={sign}
                    className={`flex items-center justify-between p-2 rounded ${
                      count >= targetCount ? 'bg-emerald-600/20' : count > 0 ? 'bg-blue-600/20' : 'bg-slate-900/50'
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-bold text-white">{sign}</span>
                      <button
                        onClick={() => removeSign(sign)}
                        className="text-red-400 hover:text-red-300 text-xs"
                        title={`Remove ${sign}`}
                      >
                        ✕
                      </button>
                    </div>
                    <div className={`text-sm font-bold ${count >= targetCount ? 'text-emerald-400' : 'text-slate-400'}`}>
                      {count}/{targetCount}
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="text-center text-slate-500 text-sm py-4">
              No signs added yet.<br/>Add a sign name to start capturing.
            </div>
          )}
        </div>

        {/* Actions */}
        <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
          <h3 className="text-sm font-bold text-slate-400 uppercase mb-3">Actions</h3>
          <div className="flex flex-col gap-2">
            <button
              onClick={downloadImages}
              disabled={capturedImages.length === 0}
              className="w-full px-4 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white rounded-lg font-bold flex items-center justify-center gap-2"
            >
              <Download className="w-4 h-4" />
              Download Dataset
            </button>
            <button
              onClick={clearCurrentSign}
              disabled={!currentSign || currentSignCount === 0}
              className="w-full px-4 py-2 bg-red-600/20 hover:bg-red-600/30 disabled:opacity-50 text-red-400 rounded-lg font-bold flex items-center justify-center gap-2"
            >
              <Trash2 className="w-4 h-4" />
              Clear "{currentSign || 'None'}" ({currentSignCount})
            </button>
          </div>
        </div>

        {/* Train Model Section */}
        <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
          <div 
            className="flex items-center justify-between cursor-pointer"
            onClick={() => setShowTrainingPanel(!showTrainingPanel)}
          >
            <h3 className="text-sm font-bold text-slate-400 uppercase flex items-center gap-2">
              <Brain className="w-4 h-4" />
              Train Model
            </h3>
            <span className="text-slate-500 text-xs">{showTrainingPanel ? '▼' : '▶'}</span>
          </div>
          
          {showTrainingPanel && (
            <div className="mt-4 space-y-3">
              {/* Model Name Input */}
              <div>
                <label className="text-xs text-slate-500 block mb-1">Model Name</label>
                <input
                  type="text"
                  value={modelName}
                  onChange={(e) => setModelName(e.target.value)}
                  placeholder="my_asl_model"
                  className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white text-sm focus:outline-none focus:border-blue-500"
                  disabled={isTraining}
                />
              </div>
              
              {/* Training Progress */}
              {isTraining && trainingProgress && (
                <div className="bg-slate-900/50 rounded-lg p-3">
                  <div className="flex justify-between text-xs text-slate-400 mb-1">
                    <span>{trainingProgress.status}</span>
                    <span>Epoch {trainingProgress.epoch}/{trainingProgress.totalEpochs}</span>
                  </div>
                  <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-gradient-to-r from-blue-500 to-emerald-500 transition-all duration-300"
                      style={{ width: `${(trainingProgress.epoch / trainingProgress.totalEpochs) * 100}%` }}
                    />
                  </div>
                  <div className="flex justify-between text-xs text-slate-500 mt-1">
                    <span>Loss: {trainingProgress.loss.toFixed(4)}</span>
                    <span>Acc: {(trainingProgress.accuracy * 100).toFixed(1)}%</span>
                  </div>
                </div>
              )}
              
              {/* Train Button */}
              <button
                onClick={handleTrainModel}
                disabled={isTraining || capturedImages.length === 0}
                className="w-full px-4 py-2 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-500 hover:to-blue-500 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-lg font-bold flex items-center justify-center gap-2"
              >
                {isTraining ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Training...
                  </>
                ) : (
                  <>
                    <Zap className="w-4 h-4" />
                    Train Model
                  </>
                )}
              </button>
              
              {/* Requirements Info */}
              <div className="text-xs text-slate-500 bg-slate-900/30 rounded p-2">
                <p className="font-bold text-slate-400 mb-1">Requirements:</p>
                <ul className="space-y-0.5 list-disc list-inside">
                  <li>At least 2 different letters</li>
                  <li>10+ images minimum</li>
                  <li>More data = better accuracy</li>
                </ul>
              </div>
              
              {/* Trained Models List */}
              {trainedModels.length > 0 && (
                <div className="mt-3">
                  <h4 className="text-xs font-bold text-slate-400 mb-2">Saved Models:</h4>
                  <div className="space-y-1 max-h-24 overflow-y-auto">
                    {trainedModels.map(m => (
                      <div 
                        key={m.name} 
                        className="flex items-center justify-between bg-slate-900/50 rounded px-2 py-1"
                      >
                        <span className="text-xs text-white truncate">{m.name}</span>
                        <span className="text-xs text-emerald-400">{(m.accuracy * 100).toFixed(0)}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Instructions */}
        <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
          <h3 className="text-sm font-bold text-slate-400 uppercase mb-2">How to Use</h3>
          <ol className="text-xs text-slate-500 space-y-1 list-decimal list-inside">
            <li>Add a sign name (e.g., "HELLO", "YES", "A")</li>
            <li>Start camera and show your hand</li>
            <li>Click Capture or use Auto Capture</li>
            <li>Capture 50+ images per sign</li>
            <li>Add more signs as needed</li>
            <li>Click "Train Model" to train</li>
            <li>Use in Sign to Text with Model Switch</li>
          </ol>
        </div>
      </div>
    </div>
  );
};

export default DatasetCapture;