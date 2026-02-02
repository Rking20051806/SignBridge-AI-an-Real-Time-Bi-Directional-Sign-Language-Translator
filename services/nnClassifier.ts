/**
 * Neural Network ASL Classifier using trained weights
 * Uses the landmark-based model trained on combined datasets
 * Achieves 91.96% test accuracy
 * Supports custom trained models
 */

interface Landmark {
  x: number;
  y: number;
  z: number;
}

// Model weights (loaded dynamically)
let CLASSES: string[] = [];
let WEIGHTS: any[] = [];
let isModelLoaded = false;
let modelLoadPromise: Promise<void> | null = null;

// Custom model support
let currentModelName = 'default';
let defaultClasses: string[] = [];
let defaultWeights: any[] = [];

// Load model weights from public folder
async function loadModelWeights(): Promise<void> {
  if (isModelLoaded) return;
  if (modelLoadPromise) return modelLoadPromise;
  
  modelLoadPromise = (async () => {
    try {
      const response = await fetch('/asl_trained_weights.json');
      const data = await response.json();
      
      CLASSES = data.classes;
      WEIGHTS = data.weights;
      
      // Store default model
      defaultClasses = [...CLASSES];
      defaultWeights = [...WEIGHTS];
      
      isModelLoaded = true;
      
      console.log('✅ ASL Neural Network Model loaded successfully');
      console.log(`   Classes: ${CLASSES.length} (${CLASSES.join(', ')})`);
      console.log(`   Layers: ${WEIGHTS.length}`);
    } catch (error) {
      console.error('Failed to load model weights:', error);
      throw error;
    }
  })();
  
  return modelLoadPromise;
}

// Switch to a custom model
export function switchToCustomModel(modelData: { classes: string[]; weights: any[]; name: string }): void {
  CLASSES = modelData.classes;
  WEIGHTS = modelData.weights;
  currentModelName = modelData.name;
  console.log(`🔄 Switched to custom model: ${modelData.name}`);
  console.log(`   Classes: ${CLASSES.length} (${CLASSES.join(', ')})`);
}

// Switch back to default model
export function switchToDefaultModel(): void {
  if (defaultClasses.length > 0 && defaultWeights.length > 0) {
    CLASSES = [...defaultClasses];
    WEIGHTS = [...defaultWeights];
    currentModelName = 'default';
    console.log('🔄 Switched back to default model');
  }
}

// Get current model name
export function getCurrentModelName(): string {
  return currentModelName;
}

// Initialize model loading
loadModelWeights();

// ============ NEURAL NETWORK MATH ============

function relu(x: number): number {
  return Math.max(0, x);
}

function softmax(arr: number[]): number[] {
  const max = Math.max(...arr);
  const exp = arr.map(x => Math.exp(x - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map(x => x / sum);
}

function denseForward(input: number[], kernel: number[][], bias: number[]): number[] {
  const output: number[] = [];
  
  for (let j = 0; j < kernel[0].length; j++) {
    let sum = bias[j];
    for (let i = 0; i < input.length; i++) {
      sum += input[i] * kernel[i][j];
    }
    output.push(sum);
  }
  
  return output;
}

function batchNormForward(input: number[], gamma: number[], beta: number[], mean: number[], variance: number[], epsilon = 1e-5): number[] {
  return input.map((x, i) => {
    const normalized = (x - mean[i]) / Math.sqrt(variance[i] + epsilon);
    return gamma[i] * normalized + beta[i];
  });
}

// ============ FEATURE EXTRACTION ============

function normalizeLandmarks(landmarks: Landmark[]): number[] {
  if (landmarks.length !== 21) return [];
  
  // Flatten to array
  const points: number[][] = landmarks.map(l => [l.x, l.y, l.z]);
  
  // Center around wrist (landmark 0)
  const wrist = points[0].slice();
  const centered = points.map(p => [p[0] - wrist[0], p[1] - wrist[1], p[2] - wrist[2]]);
  
  // Scale by distance from wrist to middle finger MCP (landmark 9)
  const middleMcp = centered[9];
  const scale = Math.sqrt(
    Math.pow(middleMcp[0], 2) +
    Math.pow(middleMcp[1], 2) +
    Math.pow(middleMcp[2], 2)
  );
  
  if (scale > 0) {
    for (const p of centered) {
      p[0] /= scale;
      p[1] /= scale;
      p[2] /= scale;
    }
  }
  
  // Flatten to 63 features
  return centered.flat();
}

// ============ NEURAL NETWORK INFERENCE ============

let lastConfidence = 0;

function neuralNetworkPredict(input: number[]): { letter: string; confidence: number } {
  if (input.length !== 63) {
    return { letter: '...', confidence: 0 };
  }
  
  let x = input;
  
  // Process through layers based on exported weights
  // New Architecture: Dense -> BN -> Activation(relu) -> Dropout (repeated) -> Dense(softmax)
  
  for (let i = 0; i < WEIGHTS.length; i++) {
    const layer = WEIGHTS[i];
    
    if (layer.type === 'Dense') {
      const kernel = layer.weights[0];
      const bias = layer.weights[1];
      x = denseForward(x, kernel, bias);
    } else if (layer.type === 'BatchNormalization') {
      // BatchNormalization weights: [gamma, beta, moving_mean, moving_variance]
      const gamma = layer.weights[0];
      const beta = layer.weights[1];
      const mean = layer.weights[2];
      const variance = layer.weights[3];
      x = batchNormForward(x, gamma, beta, mean, variance);
    } else if (layer.type === 'Activation') {
      // Apply activation function
      x = x.map(relu);
    }
    // Dropout is ignored at inference time
  }
  
  // Apply softmax to final output
  x = softmax(x);
  
  // Find max probability
  let maxIdx = 0;
  let maxProb = x[0];
  for (let i = 1; i < x.length; i++) {
    if (x[i] > maxProb) {
      maxProb = x[i];
      maxIdx = i;
    }
  }
  
  lastConfidence = maxProb;
  
  return {
    letter: CLASSES[maxIdx],
    confidence: maxProb
  };
}

// ============ EXPORTS ============

let debugLogCounter = 0;

export function classifyHandSignNN(landmarks: Landmark[] | undefined | null): string {
  if (!landmarks || landmarks.length < 21) {
    return '...';
  }
  
  // Check if model is loaded
  if (!isModelLoaded || WEIGHTS.length === 0) {
    if (debugLogCounter % 100 === 0) {
      console.log('⏳ Model not loaded yet, weights:', WEIGHTS.length);
    }
    debugLogCounter++;
    return '...';
  }
  
  try {
    // Extract normalized landmarks (63 features)
    const features = normalizeLandmarks(landmarks);
    
    if (features.length !== 63) {
      console.log('❌ Feature extraction failed:', features.length);
      return '...';
    }
    
    const result = neuralNetworkPredict(features);
    
    // Log every prediction for debugging
    if (debugLogCounter % 10 === 0) {
      console.log(`🔍 NN: ${result.letter} (${(result.confidence * 100).toFixed(1)}%)`);
    }
    debugLogCounter++;
    
    // Accept predictions with confidence > 40%
    if (result.confidence < 0.40) {
      return '...';
    }
    
    return result.letter;
  } catch (e) {
    console.error('NN Classification error:', e);
    return '...';
  }
}

export function getConfidenceNN(): number {
  return lastConfidence;
}

export function getClassesNN(): string[] {
  return CLASSES;
}

export function isNNModelLoaded(): boolean {
  return isModelLoaded;
}

export async function waitForModelLoad(): Promise<void> {
  return loadModelWeights();
}
