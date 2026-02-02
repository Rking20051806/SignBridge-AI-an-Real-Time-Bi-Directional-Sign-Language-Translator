import React, { useRef, useState, useEffect, useCallback } from 'react';
import { Video, StopCircle, Play, Loader2, LayoutGrid, Scan, Image as ImageIcon, CheckCircle2, Fingerprint, AlertCircle, Maximize2, User, Wifi, Activity, Zap, Search, Aperture, Target, Settings2, Gauge, ShieldCheck, ShieldAlert, XCircle, Cpu, RefreshCw } from 'lucide-react';
import { classifyHandSignNN, getConfidenceNN, isNNModelLoaded, waitForModelLoad, switchToCustomModel, switchToDefaultModel, getCurrentModelName } from '../services/nnClassifier';
import { getCustomModels, loadCustomModel } from '../services/modelTrainer';
import { Detection } from '../types';
import { ASL_ALPHABET } from '../constants';
import { HandLandmarker, FilesetResolver, PoseLandmarker } from "@mediapipe/tasks-vision";

// Connection maps with finger colors
// Thumb: Red, Index: Green, Middle: Blue, Ring: Yellow, Pinky: Magenta, Palm: Cyan
const FINGER_COLORS = {
  thumb: "#FF4444",    // Red
  index: "#44FF44",    // Green  
  middle: "#4444FF",   // Blue
  ring: "#FFFF44",     // Yellow
  pinky: "#FF44FF",    // Magenta
  palm: "#44FFFF"      // Cyan
};

const HAND_CONNECTIONS_COLORED = [
  // Thumb (Red)
  {start: 0, end: 1, finger: 'thumb'}, {start: 1, end: 2, finger: 'thumb'}, 
  {start: 2, end: 3, finger: 'thumb'}, {start: 3, end: 4, finger: 'thumb'},
  // Index (Green)
  {start: 0, end: 5, finger: 'palm'}, {start: 5, end: 6, finger: 'index'}, 
  {start: 6, end: 7, finger: 'index'}, {start: 7, end: 8, finger: 'index'},
  // Middle (Blue)
  {start: 5, end: 9, finger: 'palm'}, {start: 9, end: 10, finger: 'middle'}, 
  {start: 10, end: 11, finger: 'middle'}, {start: 11, end: 12, finger: 'middle'},
  // Ring (Yellow)
  {start: 9, end: 13, finger: 'palm'}, {start: 13, end: 14, finger: 'ring'}, 
  {start: 14, end: 15, finger: 'ring'}, {start: 15, end: 16, finger: 'ring'},
  // Pinky (Magenta)
  {start: 13, end: 17, finger: 'palm'}, {start: 0, end: 17, finger: 'palm'}, 
  {start: 17, end: 18, finger: 'pinky'}, {start: 18, end: 19, finger: 'pinky'}, 
  {start: 19, end: 20, finger: 'pinky'}
];

// Landmark to finger mapping for colored dots
const LANDMARK_FINGERS: {[key: number]: string} = {
  0: 'palm',  // Wrist
  1: 'thumb', 2: 'thumb', 3: 'thumb', 4: 'thumb',  // Thumb
  5: 'palm', 6: 'index', 7: 'index', 8: 'index',   // Index
  9: 'palm', 10: 'middle', 11: 'middle', 12: 'middle', // Middle
  13: 'palm', 14: 'ring', 15: 'ring', 16: 'ring',  // Ring
  17: 'palm', 18: 'pinky', 19: 'pinky', 20: 'pinky' // Pinky
};

const HAND_CONNECTIONS = [
  {start: 0, end: 1}, {start: 1, end: 2}, {start: 2, end: 3}, {start: 3, end: 4},
  {start: 0, end: 5}, {start: 5, end: 6}, {start: 6, end: 7}, {start: 7, end: 8},
  {start: 5, end: 9}, {start: 9, end: 10}, {start: 10, end: 11}, {start: 11, end: 12},
  {start: 9, end: 13}, {start: 13, end: 14}, {start: 14, end: 15}, {start: 15, end: 16},
  {start: 13, end: 17}, {start: 0, end: 17}, {start: 17, end: 18}, {start: 18, end: 19}, {start: 19, end: 20}
];

const POSE_CONNECTIONS = [
  {start: 11, end: 12}, {start: 11, end: 13}, {start: 13, end: 15}, 
  {start: 12, end: 14}, {start: 14, end: 16}
];

type InputMode = 'webcam' | 'ip-cam';

const SignToText: React.FC = () => {
  const [inputMode, setInputMode] = useState<InputMode>('webcam');
  const [ipCamUrl, setIpCamUrl] = useState<string>('http://192.168.1.5:4747/video');
  
  // Settings State
  const [stabilization, setStabilization] = useState(true);
  const [captureInterval, setCaptureInterval] = useState(100); // Fast local processing
  const [showSettings, setShowSettings] = useState(false);
  const [confidence, setConfidence] = useState(0);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null); 
  const cropCanvasRef = useRef<HTMLCanvasElement>(null);
  
  // MediaPipe - Using HandLandmarker for faster loading
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const poseLandmarkerRef = useRef<PoseLandmarker | null>(null);
  const requestRef = useRef<number>(0);
  const lastVideoTimeRef = useRef<number>(-1);
  const [loadingProgress, setLoadingProgress] = useState<string>('Initializing...');
  
  // Logic State
  const activeLandmarksRef = useRef<any>(null);
  const predictionBufferRef = useRef<string[]>([]); // For stabilization
  
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [currentSign, setCurrentSign] = useState<string>("...");
  const [detections, setDetections] = useState<Detection[]>([]);
  const [history, setHistory] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [matchedLetter, setMatchedLetter] = useState<string | null>(null);
  const [handPresent, setHandPresent] = useState(false);
  
  // Model selection state
  const [availableModels, setAvailableModels] = useState<{ name: string; trainedAt: string; accuracy: number }[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('default');
  const [showModelSelector, setShowModelSelector] = useState(false);
  
  const intervalRef = useRef<number | null>(null);

  // Load available custom models on mount
  useEffect(() => {
    const models = getCustomModels();
    setAvailableModels(models);
    console.log(`📦 Found ${models.length} custom trained models`);
  }, []);

  useEffect(() => {
    return () => stopCamera();
  }, []);

  useEffect(() => {
    const loadMediaPipe = async () => {
        try {
            setLoadingProgress('Loading vision engine...');
            
            // Load vision WASM files
            const vision = await FilesetResolver.forVisionTasks(
                "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
            );
            
            setLoadingProgress('Loading hand detection model...');
            
            // Load models in parallel for faster startup
            const [handLandmarker, nnModel] = await Promise.all([
                // Hand Landmarker - much smaller and faster than Holistic
                HandLandmarker.createFromOptions(vision, {
                    baseOptions: {
                        modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
                        delegate: "GPU"
                    },
                    runningMode: "VIDEO",
                    numHands: 2,
                    minHandDetectionConfidence: 0.5,
                    minHandPresenceConfidence: 0.5,
                    minTrackingConfidence: 0.5
                }),
                // Load NN weights in parallel
                waitForModelLoad()
            ]);
            
            handLandmarkerRef.current = handLandmarker;
            
            setLoadingProgress('Ready!');
            console.log('✅ All models loaded successfully');
            setIsModelLoading(false);
        } catch (err: any) {
            console.error(err);
            setError("Vision Module Failed to Load. Check internet connection.");
            setIsModelLoading(false);
        }
    };
    loadMediaPipe();
  }, []);


  const startCamera = async () => {
    setError(null);
    if (!videoRef.current) return;

    // Check if MediaPipe model is loaded
    if (!handLandmarkerRef.current) {
      setError("Model is still loading. Please wait...");
      return;
    }

    try {
      if (inputMode === 'webcam') {
        const stream = await navigator.mediaDevices.getUserMedia({ 
          video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" } 
        });
        videoRef.current.srcObject = stream;
      } else if (inputMode === 'ip-cam') {
        videoRef.current.crossOrigin = "anonymous";
        videoRef.current.src = ipCamUrl;
      }

      // Wait for video to be ready before starting prediction loop
      await new Promise<void>((resolve, reject) => {
        const video = videoRef.current!;
        
        const onLoadedData = () => {
          video.removeEventListener('loadeddata', onLoadedData);
          video.removeEventListener('error', onError);
          resolve();
        };
        
        const onError = () => {
          video.removeEventListener('loadeddata', onLoadedData);
          video.removeEventListener('error', onError);
          reject(new Error("Failed to load video"));
        };
        
        // If video is already loaded, resolve immediately
        if (video.readyState >= 2) {
          resolve();
        } else {
          video.addEventListener('loadeddata', onLoadedData);
          video.addEventListener('error', onError);
        }
      });

      await videoRef.current.play();
      
      // Reset the last video time to ensure detection starts fresh
      lastVideoTimeRef.current = -1;
      
      setIsStreaming(true);
      
      // Start prediction loop directly after video is ready
      requestRef.current = requestAnimationFrame(predictLoop);
    } catch (err: any) {
      setError("Camera Access Denied. Please allow permission or check device.");
    }
  };

  const stopCamera = () => {
    // Cancel animation frame first to stop the loop
    if (requestRef.current) {
        cancelAnimationFrame(requestRef.current);
        requestRef.current = 0;
    }
    
    if (videoRef.current) {
        if (videoRef.current.srcObject) {
            (videoRef.current.srcObject as MediaStream).getTracks().forEach(track => track.stop());
            videoRef.current.srcObject = null;
        }
        videoRef.current.src = '';
        videoRef.current.pause();
    }
    
    setIsStreaming(false);
    setIsAnalyzing(false);
    clearCanvas();
    setHandPresent(false);
    activeLandmarksRef.current = null;
    predictionBufferRef.current = [];
    lastVideoTimeRef.current = -1; // Reset for next session
  };

  const predictLoop = () => {
    // Check if we should continue the loop
    if (!videoRef.current || !overlayCanvasRef.current || videoRef.current.paused || videoRef.current.ended) {
        return;
    }
    
    if (handLandmarkerRef.current && videoRef.current.readyState >= 2) {
        const startTimeMs = performance.now();
        if (videoRef.current.currentTime !== lastVideoTimeRef.current) {
            lastVideoTimeRef.current = videoRef.current.currentTime;
            try {
                const result = handLandmarkerRef.current.detectForVideo(videoRef.current, startTimeMs);
                
                // HandLandmarker returns landmarks array and handedness
                const hands = result.landmarks || [];
                const handedness = result.handednesses || [];
                
                // Map to left/right based on handedness
                let left = null;
                let right = null;
                
                for (let i = 0; i < hands.length; i++) {
                    const hand = hands[i];
                    const category = handedness[i]?.[0]?.categoryName?.toLowerCase();
                    // Note: MediaPipe flips left/right in mirrored video
                    if (category === 'left') {
                        right = hand; // Mirrored
                    } else if (category === 'right') {
                        left = hand; // Mirrored
                    }
                }

                const hasHands = hands.length > 0;
                setHandPresent(hasHands);
                
                // Store landmarks for the cropper to use
                activeLandmarksRef.current = { left, right, pose: null };

                drawHandLandmarks(result);
            } catch (e) { 
                console.warn("Detection error:", e);
            }
        }
    }

    // Continue the loop
    requestRef.current = requestAnimationFrame(predictLoop);
  };

  const drawHandLandmarks = (result: any) => {
    const canvas = overlayCanvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!ctx || !canvas || !videoRef.current) return;

    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw Hands & Bounding Box with colored fingers
    // HandLandmarker returns landmarks as array of hands
    const hands = result.landmarks || [];
    
    hands.forEach((hand: any) => {
        if (hand && hand.length > 0) {
            // Draw colored connections for each finger
            drawColoredConnectors(ctx, hand, HAND_CONNECTIONS_COLORED);
            
            // Draw colored landmarks for each finger
            drawColoredLandmarks(ctx, hand);
            
            const { minX, minY, width, height } = getBoundingBox(hand, canvas.width, canvas.height);
            ctx.strokeStyle = "#34d399";
            ctx.lineWidth = 2;
            
            const lineLen = 20;
            ctx.beginPath(); ctx.moveTo(minX, minY + lineLen); ctx.lineTo(minX, minY); ctx.lineTo(minX + lineLen, minY); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(minX + width - lineLen, minY); ctx.lineTo(minX + width, minY); ctx.lineTo(minX + width, minY + lineLen); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(minX, minY + height - lineLen); ctx.lineTo(minX, minY + height); ctx.lineTo(minX + lineLen, minY + height); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(minX + width - lineLen, minY + height); ctx.lineTo(minX + width, minY + height); ctx.lineTo(minX + width, minY + height - lineLen); ctx.stroke();
        }
    });
  };

  // Keep for potential fallback use
  const drawHolisticLandmarks = (result: any) => {
    const canvas = overlayCanvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!ctx || !canvas || !videoRef.current) return;

    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw Pose
    if (result.poseLandmarks?.[0]) {
        drawConnectors(ctx, result.poseLandmarks[0], POSE_CONNECTIONS, { 
            color: "rgba(56, 189, 248, 0.6)", 
            lineWidth: 6 
        });
        drawLandmarks(ctx, result.poseLandmarks[0], { 
            color: "rgba(56, 189, 248, 1)", 
            radius: 6 
        });
    }

    // Draw Hands & Bounding Box with colored fingers
    [result.leftHandLandmarks?.[0], result.rightHandLandmarks?.[0]].forEach(hand => {
        if (hand) {
            // Draw colored connections for each finger
            drawColoredConnectors(ctx, hand, HAND_CONNECTIONS_COLORED);
            
            // Draw colored landmarks for each finger
            drawColoredLandmarks(ctx, hand);
            
            const { minX, minY, width, height } = getBoundingBox(hand, canvas.width, canvas.height);
            ctx.strokeStyle = "#34d399";
            ctx.lineWidth = 2;
            
            const lineLen = 20;
            ctx.beginPath(); ctx.moveTo(minX, minY + lineLen); ctx.lineTo(minX, minY); ctx.lineTo(minX + lineLen, minY); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(minX + width - lineLen, minY); ctx.lineTo(minX + width, minY); ctx.lineTo(minX + width, minY + lineLen); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(minX, minY + height - lineLen); ctx.lineTo(minX, minY + height); ctx.lineTo(minX + lineLen, minY + height); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(minX + width - lineLen, minY + height); ctx.lineTo(minX + width, minY + height); ctx.lineTo(minX + width, minY + height - lineLen); ctx.stroke();
        }
    });
  };

  // Draw connections with different colors per finger
  const drawColoredConnectors = (ctx: CanvasRenderingContext2D, landmarks: any[], connections: any[]) => {
    ctx.lineWidth = 6;
    for (const c of connections) {
        const f = landmarks[c.start];
        const t = landmarks[c.end];
        const color = FINGER_COLORS[c.finger as keyof typeof FINGER_COLORS] || "#FFFFFF";
        
        ctx.strokeStyle = color;
        ctx.beginPath();
        ctx.moveTo(f.x * ctx.canvas.width, f.y * ctx.canvas.height);
        ctx.lineTo(t.x * ctx.canvas.width, t.y * ctx.canvas.height);
        ctx.stroke();
    }
  };

  // Draw landmarks with different colors per finger
  const drawColoredLandmarks = (ctx: CanvasRenderingContext2D, landmarks: any[]) => {
    landmarks.forEach((l: any, idx: number) => {
        const finger = LANDMARK_FINGERS[idx] || 'palm';
        const color = FINGER_COLORS[finger as keyof typeof FINGER_COLORS] || "#FFFFFF";
        
        // Outer ring (white border)
        ctx.fillStyle = "#FFFFFF";
        ctx.beginPath();
        ctx.arc(l.x * ctx.canvas.width, l.y * ctx.canvas.height, 10, 0, 2 * Math.PI);
        ctx.fill();
        
        // Inner colored dot
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(l.x * ctx.canvas.width, l.y * ctx.canvas.height, 7, 0, 2 * Math.PI);
        ctx.fill();
        
        // Fingertip labels (for tips only: 4, 8, 12, 16, 20)
        if ([4, 8, 12, 16, 20].includes(idx)) {
            const labels: {[key: number]: string} = {4: 'T', 8: 'I', 12: 'M', 16: 'R', 20: 'P'};
            ctx.fillStyle = "#000000";
            ctx.font = "bold 10px Arial";
            ctx.textAlign = "center";
            ctx.fillText(labels[idx], l.x * ctx.canvas.width, l.y * ctx.canvas.height + 3);
        }
    });
  };

  const getBoundingBox = (landmarks: any[], width: number, height: number) => {
      const x = landmarks.map(l => l.x);
      const y = landmarks.map(l => l.y);
      const minX = Math.min(...x) * width;
      const maxX = Math.max(...x) * width;
      const minY = Math.min(...y) * height;
      const maxY = Math.max(...y) * height;
      
      const padding = 40;
      return {
          minX: Math.max(0, minX - padding),
          minY: Math.max(0, minY - padding),
          width: Math.min(width, (maxX - minX) + (padding * 2)),
          height: Math.min(height, (maxY - minY) + (padding * 2))
      };
  };

  const drawConnectors = (ctx: CanvasRenderingContext2D, landmarks: any[], connections: any[], style: any) => {
      ctx.strokeStyle = style.color;
      ctx.lineWidth = style.lineWidth;
      ctx.beginPath();
      for (const c of connections) {
          const f = landmarks[c.start];
          const t = landmarks[c.end];
          ctx.moveTo(f.x * ctx.canvas.width, f.y * ctx.canvas.height);
          ctx.lineTo(t.x * ctx.canvas.width, t.y * ctx.canvas.height);
      }
      ctx.stroke();
  };

  const drawLandmarks = (ctx: CanvasRenderingContext2D, landmarks: any[], style: any) => {
      ctx.fillStyle = style.color;
      for (const l of landmarks) {
          ctx.beginPath();
          ctx.arc(l.x * ctx.canvas.width, l.y * ctx.canvas.height, style.radius, 0, 2 * Math.PI);
          ctx.fill();
      }
  };

  const clearCanvas = () => {
    const ctx = overlayCanvasRef.current?.getContext('2d');
    ctx?.clearRect(0, 0, overlayCanvasRef.current?.width || 0, overlayCanvasRef.current?.height || 0);
  };

  // Analysis Loop - Now using trained neural network classifier
  const captureAndAnalyze = useCallback(async () => {
    if (!videoRef.current || !isStreaming) return; 
    
    // Check if NN model is loaded
    if (!isNNModelLoaded()) {
      console.log('⏳ Waiting for NN model to load...');
      return;
    }
    
    // Get hand landmarks directly from MediaPipe results
    const landmarks = activeLandmarksRef.current;
    
    if (!landmarks || (!landmarks.left && !landmarks.right)) {
      return; // Skip if no hands detected
    }

    try {
      // Use the dominant/visible hand for classification
      const handLandmarks = landmarks.right || landmarks.left;
      
      // Classify using trained neural network
      const result = classifyHandSignNN(handLandmarks);
      
      if (!result || result === "...") return;

      const cleanText = result.trim().toUpperCase();
      
      // Check if it's a valid ASL letter or space
      const isValidSign = (cleanText.length === 1 && ASL_ALPHABET[cleanText.toLowerCase()]) || cleanText.toLowerCase() === 'space';
      if (isValidSign) {
          let finalSign = cleanText === 'SPACE' ? ' ' : cleanText;
          
          // Get confidence score from NN
          const conf = getConfidenceNN();
          setConfidence(Math.round(conf * 100));

          // Stabilization Logic
          if (stabilization) {
              const buffer = predictionBufferRef.current;
              buffer.push(cleanText);
              if (buffer.length > 5) buffer.shift(); // Keep last 5 frames for smoother results
              
              // Majority Vote
              const counts = buffer.reduce((acc: any, val) => { acc[val] = (acc[val] || 0) + 1; return acc; }, {});
              const maxCount = Math.max(...Object.values(counts) as number[]);
              const candidate = Object.keys(counts).find(key => counts[key] === maxCount);

              // Require at least 2 consistent frames for faster detection
              if (maxCount < 2) return;
              finalSign = candidate === 'SPACE' ? ' ' : candidate!;
          }

          setCurrentSign(prev => {
              if (prev !== finalSign) {
                  setHistory(h => [finalSign, ...h].slice(0, 8));
                  return finalSign;
              }
              return prev;
          });
          setMatchedLetter(cleanText === 'SPACE' ? 'space' : finalSign.toLowerCase());
          setError(null);
      }
    } catch (err: any) { 
        console.error(err);
    } 
  }, [isStreaming, stabilization]);

  // Manage Analysis Loop based on state - must be after captureAndAnalyze definition
  useEffect(() => {
      if (isAnalyzing && isStreaming) {
          if (intervalRef.current) clearInterval(intervalRef.current);
          intervalRef.current = window.setInterval(captureAndAnalyze, captureInterval);
      } else {
          if (intervalRef.current) clearInterval(intervalRef.current);
          intervalRef.current = null;
      }
      return () => {
          if (intervalRef.current) clearInterval(intervalRef.current);
      };
  }, [isAnalyzing, isStreaming, captureInterval, stabilization, captureAndAnalyze]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-3 sm:gap-6 h-full">
      {/* Main Interface */}
      <div className="lg:col-span-8 flex flex-col gap-3 sm:gap-4">
        
        {/* Control Bar */}
        <div className="bg-slate-900/80 p-1 sm:p-1.5 rounded-lg sm:rounded-xl flex items-center justify-between border border-slate-700/50 backdrop-blur-md relative z-20">
          <div className="flex gap-1">
            <div className="px-2 sm:px-4 py-1.5 sm:py-2 rounded-lg text-[10px] sm:text-xs font-bold uppercase tracking-wider bg-slate-700 text-white shadow-inner">
              Live Feed
            </div>
          </div>
          
          <div className="flex items-center gap-2 sm:gap-3 px-1 sm:px-2">
            {isStreaming && (
              <div className="flex items-center gap-1 sm:gap-2 text-[8px] sm:text-[10px] font-mono text-emerald-400 mr-1 sm:mr-2">
                <Target className={`w-2.5 h-2.5 sm:w-3 sm:h-3 ${handPresent ? 'animate-ping' : 'opacity-20'}`} />
                <span className="hidden sm:inline">{handPresent ? "LOCKED" : "SCAN"}</span>
              </div>
            )}
            
            {/* Settings Toggle */}
            <button 
              onClick={() => setShowSettings(!showSettings)}
              className={`p-1.5 sm:p-2 rounded-lg transition-all ${showSettings ? 'bg-blue-600 text-white' : 'text-slate-400 hover:bg-slate-800'}`}
              title="Settings"
            >
              <Settings2 className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
            </button>
          </div>

          {/* Settings Popover */}
          {showSettings && (
            <div className="absolute top-full right-0 mt-2 w-64 bg-slate-800 border border-slate-700 rounded-xl shadow-xl p-4 flex flex-col gap-4 animate-in slide-in-from-top-2 z-50">
              <div className="flex items-center gap-2 text-emerald-400 text-xs pb-2 border-b border-slate-700">
                <Cpu className="w-3 h-3" />
                <span className="font-bold">LOCAL PROCESSING (NO API)</span>
              </div>
              
              <div className="space-y-2">
                <label className="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center gap-2">
                  <Gauge className="w-3 h-3" /> Detection Speed
                </label>
                <div className="grid grid-cols-3 gap-1 bg-slate-900 p-1 rounded-lg">
                  <button onClick={() => setCaptureInterval(200)} className={`text-[10px] font-bold py-1 rounded ${captureInterval === 200 ? 'bg-slate-700 text-white' : 'text-slate-500 hover:text-slate-300'}`}>SLOW</button>
                  <button onClick={() => setCaptureInterval(100)} className={`text-[10px] font-bold py-1 rounded ${captureInterval === 100 ? 'bg-slate-700 text-white' : 'text-slate-500 hover:text-slate-300'}`}>NORM</button>
                  <button onClick={() => setCaptureInterval(50)} className={`text-[10px] font-bold py-1 rounded ${captureInterval === 50 ? 'bg-slate-700 text-white' : 'text-slate-500 hover:text-slate-300'}`}>FAST</button>
                </div>
              </div>

              <div className="space-y-2">
                <label className="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center gap-2">
                  {stabilization ? <ShieldCheck className="w-3 h-3 text-emerald-400" /> : <ShieldAlert className="w-3 h-3 text-amber-400" />} 
                  Stabilization
                </label>
                <button 
                  onClick={() => setStabilization(!stabilization)}
                  className={`w-full py-2 rounded-lg text-xs font-bold border transition-all ${stabilization ? 'bg-emerald-500/10 border-emerald-500/50 text-emerald-400' : 'bg-slate-900 border-slate-700 text-slate-500'}`}
                >
                  {stabilization ? "ENABLED (SMOOTH)" : "DISABLED (RAW)"}
                </button>
              </div>

              {/* Model Selector - Only show if custom models exist */}
              {availableModels.length > 0 && (
                <div className="space-y-2 border-t border-slate-700 pt-3 mt-3">
                  <label className="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center gap-2">
                    <RefreshCw className="w-3 h-3" /> Model Selection
                  </label>
                  <select
                    value={selectedModel}
                    onChange={(e) => {
                      const modelName = e.target.value;
                      setSelectedModel(modelName);
                      
                      if (modelName === 'default') {
                        switchToDefaultModel();
                      } else {
                        const customModel = loadCustomModel(modelName);
                        if (customModel) {
                          switchToCustomModel({
                            name: customModel.name,
                            classes: customModel.classes,
                            weights: customModel.weights
                          });
                        }
                      }
                      
                      // Clear prediction buffer on model switch
                      predictionBufferRef.current = [];
                    }}
                    className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white text-xs focus:outline-none focus:border-blue-500"
                  >
                    <option value="default">Default Model (26 letters)</option>
                    {availableModels.map(model => (
                      <option key={model.name} value={model.name}>
                        {model.name} ({(model.accuracy * 100).toFixed(0)}%)
                      </option>
                    ))}
                  </select>
                  <p className="text-[10px] text-slate-500">
                    Using: <span className="text-emerald-400 font-bold">{selectedModel === 'default' ? 'Default' : selectedModel}</span>
                  </p>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Error Banner */}
        {error && (
          <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-3 flex items-center justify-between animate-in fade-in slide-in-from-top-2">
            <div className="flex items-center gap-2 text-red-400 text-xs font-bold">
              <AlertCircle className="w-4 h-4" />
              {error}
            </div>
            <button onClick={() => setError(null)} className="text-red-400 hover:text-red-300" title="Close"><XCircle className="w-4 h-4" /></button>
          </div>
        )}

        {/* Video Stage */}
        <div className="relative rounded-lg sm:rounded-xl overflow-hidden bg-black aspect-video shadow-2xl border border-slate-800 ring-1 ring-slate-700/50 group">
          <div className="relative w-full h-full flex items-center justify-center">
            <video ref={videoRef} className={`w-full h-full object-contain transform ${inputMode === 'webcam' ? 'scale-x-[-1]' : ''} ${isStreaming ? 'opacity-100' : 'opacity-30'}`} crossOrigin="anonymous" playsInline />
            <canvas ref={overlayCanvasRef} className={`absolute inset-0 w-full h-full pointer-events-none transform ${inputMode === 'webcam' ? 'scale-x-[-1]' : ''} object-contain`} />
            
            {/* Status Overlay */}
            <div className="absolute top-2 sm:top-4 left-2 sm:left-4 flex gap-1 sm:gap-2">
              <div className="bg-black/50 backdrop-blur px-1.5 sm:px-2 py-0.5 sm:py-1 rounded border border-white/10 text-[8px] sm:text-[10px] text-white/70 font-mono">
                <span className="hidden sm:inline">CAM_01: </span>{isStreaming ? "ONLINE" : "OFF"}
              </div>
              {isAnalyzing && (
                <div className="bg-red-500/20 backdrop-blur px-1.5 sm:px-2 py-0.5 sm:py-1 rounded border border-red-500/30 text-[8px] sm:text-[10px] text-red-400 font-mono animate-pulse">
                  REC
                </div>
              )}
            </div>

            {/* Scanner Line Effect */}
            {isAnalyzing && (
              <div className="absolute inset-0 bg-gradient-to-b from-transparent via-emerald-500/5 to-transparent h-[20%] animate-scan pointer-events-none z-10" />
            )}

            {!isStreaming && (
              <div className="absolute inset-0 flex flex-col items-center justify-center p-4">
                <Aperture className="w-12 h-12 sm:w-16 sm:h-16 text-slate-700 mb-3 sm:mb-4 opacity-50" />
                <button onClick={startCamera} disabled={isModelLoading} className="px-5 sm:px-8 py-2.5 sm:py-3 bg-blue-600 hover:bg-blue-500 text-white rounded-full font-bold shadow-lg shadow-blue-500/25 transition-all flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed text-sm sm:text-base">
                  {isModelLoading ? <Loader2 className="animate-spin w-4 h-4"/> : <Zap className="w-4 h-4" />}
                  {isModelLoading ? loadingProgress : "ACTIVATE"}
                </button>
                {isModelLoading && (
                  <p className="text-[10px] sm:text-xs text-slate-500 mt-2 text-center">Using lightweight hand detection model...</p>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Action Bar */}
        {isStreaming && (
          <div className="grid grid-cols-2 gap-2 sm:gap-4">
            <button 
              onClick={() => setIsAnalyzing(!isAnalyzing)} 
              className={`py-3 sm:py-4 rounded-lg sm:rounded-xl font-bold sm:font-black uppercase tracking-wide sm:tracking-widest transition-all flex items-center justify-center gap-2 sm:gap-3 shadow-lg text-xs sm:text-base ${isAnalyzing ? 'bg-amber-500/10 text-amber-500 border border-amber-500/50 hover:bg-amber-500/20' : 'bg-emerald-600 text-white hover:bg-emerald-500 shadow-emerald-500/20'}`}
            >
              {isAnalyzing ? <StopCircle className="w-4 h-4 sm:w-5 sm:h-5" /> : <Play className="w-4 h-4 sm:w-5 sm:h-5" />}
              <span className="hidden sm:inline">{isAnalyzing ? "PAUSE INFERENCE" : "START TRANSLATION"}</span>
              <span className="sm:hidden">{isAnalyzing ? "PAUSE" : "START"}</span>
            </button>
            
            <button onClick={stopCamera} className="py-3 sm:py-4 bg-slate-800 text-slate-400 hover:text-white hover:bg-slate-700 rounded-lg sm:rounded-xl font-bold uppercase tracking-wide sm:tracking-widest border border-slate-700 text-xs sm:text-base">
              <span className="hidden sm:inline">TERMINATE</span>
              <span className="sm:hidden">STOP</span>
            </button>
          </div>
        )}
        
        {/* Hidden Crop Canvas */}
        <canvas ref={cropCanvasRef} className="hidden" />
      </div>

      {/* Info / Result Sidebar */}
      <div className="lg:col-span-4 flex flex-col gap-3 sm:gap-4">
        
        {/* Main Result Display */}
        <div className="bg-slate-800 border border-slate-700 rounded-lg sm:rounded-xl p-1 shadow-2xl relative overflow-hidden group">
          <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-blue-500 via-purple-500 to-emerald-500 opacity-50 group-hover:opacity-100 transition-opacity"></div>
          
          <div className="bg-slate-900/50 p-4 sm:p-6 rounded-lg min-h-[160px] sm:min-h-[240px] flex flex-col items-center justify-center relative">
            <h3 className="absolute top-2 sm:top-4 left-2 sm:left-4 text-[8px] sm:text-[10px] font-black uppercase tracking-[0.2em] text-slate-500">
              Detected Symbol
            </h3>
            
            <div className="absolute top-2 sm:top-4 right-2 sm:right-4 flex flex-col items-end gap-1">
              <div className="flex items-center gap-1 sm:gap-2 text-[8px] sm:text-[10px] text-emerald-400">
                <Cpu className="w-2.5 h-2.5 sm:w-3 sm:h-3" />
                <span className="font-mono">OFFLINE</span>
              </div>
              {selectedModel !== 'default' && (
                <div className="flex items-center gap-1 text-[8px] sm:text-[10px] text-purple-400 bg-purple-500/10 px-1.5 sm:px-2 py-0.5 rounded">
                  <RefreshCw className="w-2 h-2 sm:w-2.5 sm:h-2.5" />
                  <span className="font-mono truncate max-w-[60px] sm:max-w-none">{selectedModel}</span>
                </div>
              )}
            </div>
            
            <div className="text-6xl sm:text-8xl font-black text-transparent bg-clip-text bg-gradient-to-br from-white to-slate-400 drop-shadow-2xl">
              {currentSign}
            </div>
            
            {matchedLetter && confidence > 0 && (
              <div className="mt-4 w-full max-w-[200px]">
                <div className="flex justify-between text-[10px] text-slate-500 mb-1">
                  <span>Confidence</span>
                  <span>{Math.round(confidence * 100)}%</span>
                </div>
                <div className="h-1 bg-slate-700 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-gradient-to-r from-emerald-500 to-emerald-400 transition-all duration-300"
                    style={{ width: `${confidence * 100}%` }}
                  />
                </div>
              </div>
            )}
            
            {matchedLetter && (
              <div className="absolute bottom-4 right-4 flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div>
                <span className="text-emerald-500 text-xs font-bold tracking-widest">CONFIRMED</span>
              </div>
            )}
          </div>
        </div>

        {/* Reference Comparison */}
        <div className="bg-slate-800 border border-slate-700 rounded-xl p-4 flex-1 flex flex-col">
          <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4 flex items-center gap-2">
            <Search className="w-3 h-3" /> Database Match
          </h3>
          
          <div className="flex-1 bg-black/40 rounded-lg border-2 border-dashed border-slate-700/50 flex items-center justify-center overflow-hidden relative">
            {matchedLetter ? (
              <div className="relative w-full h-full p-4 flex items-center justify-center animate-in zoom-in duration-300">
                <img 
                  src={ASL_ALPHABET[matchedLetter]} 
                  className="max-w-full max-h-[160px] object-contain drop-shadow-[0_0_15px_rgba(16,185,129,0.3)]"
                  alt="Reference"
                />
                <div className="absolute bottom-2 left-2 bg-slate-900/80 px-2 py-1 rounded text-[10px] text-slate-300 font-mono">
                  REF_IMG_{matchedLetter.toUpperCase()}.PNG
                </div>
              </div>
            ) : (
              <div className="text-center opacity-20">
                <Fingerprint className="w-12 h-12 mx-auto mb-2" />
                <p className="text-[10px] font-mono">NO MATCH FOUND</p>
              </div>
            )}
          </div>
        </div>

        {/* Log */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 h-[200px] overflow-hidden flex flex-col">
          <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-2">Translation Log</h3>
          <div className="flex-1 overflow-y-auto space-y-1 custom-scrollbar">
            {history.map((h, i) => (
              <div key={i} className="flex items-center gap-3 text-sm font-mono text-slate-300 border-b border-slate-800/50 pb-1">
                <span className="text-slate-600 text-[10px]">{new Date().toLocaleTimeString([], {hour12:false, hour:'2-digit', minute:'2-digit', second:'2-digit'})}</span>
                <span className="text-emerald-400 font-bold">»</span>
                {h}
              </div>
            ))}
          </div>
        </div>

      </div>
    </div>
  );
};

export default SignToText;
