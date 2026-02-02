/**
 * Browser-based ASL Model Trainer using TensorFlow.js
 * Trains on captured landmark data and exports weights for the nnClassifier
 */

interface TrainingData {
  letter: string;
  landmarks: number[][];
}

export interface TrainingProgress {
  epoch: number;
  totalEpochs: number;
  loss: number;
  accuracy: number;
  status: string;
}

export interface TrainedModel {
  name: string;
  classes: string[];
  weights: any[];
  trainedAt: string;
  accuracy: number;
  totalSamples: number;
}

// Store custom models in localStorage
const CUSTOM_MODELS_KEY = 'asl_custom_models';

// Get list of available custom models
export function getCustomModels(): { name: string; trainedAt: string; accuracy: number }[] {
  try {
    const stored = localStorage.getItem(CUSTOM_MODELS_KEY);
    if (!stored) return [];
    const models = JSON.parse(stored) as TrainedModel[];
    return models.map(m => ({
      name: m.name,
      trainedAt: m.trainedAt,
      accuracy: m.accuracy
    }));
  } catch {
    return [];
  }
}

// Load a specific custom model
export function loadCustomModel(name: string): TrainedModel | null {
  try {
    const stored = localStorage.getItem(CUSTOM_MODELS_KEY);
    if (!stored) return null;
    const models = JSON.parse(stored) as TrainedModel[];
    return models.find(m => m.name === name) || null;
  } catch {
    return null;
  }
}

// Delete a custom model
export function deleteCustomModel(name: string): boolean {
  try {
    const stored = localStorage.getItem(CUSTOM_MODELS_KEY);
    if (!stored) return false;
    const models = JSON.parse(stored) as TrainedModel[];
    const filtered = models.filter(m => m.name !== name);
    localStorage.setItem(CUSTOM_MODELS_KEY, JSON.stringify(filtered));
    return true;
  } catch {
    return false;
  }
}

// Save a trained model
function saveCustomModel(model: TrainedModel): void {
  try {
    const stored = localStorage.getItem(CUSTOM_MODELS_KEY);
    let models: TrainedModel[] = stored ? JSON.parse(stored) : [];
    
    // Replace if same name exists
    models = models.filter(m => m.name !== model.name);
    models.push(model);
    
    localStorage.setItem(CUSTOM_MODELS_KEY, JSON.stringify(models));
  } catch (e) {
    console.error('Failed to save model:', e);
    throw new Error('Failed to save model to storage');
  }
}

// Check if captured dataset exists
export function hasCapturedDataset(): boolean {
  try {
    const stored = localStorage.getItem('asl_captured_dataset');
    if (!stored) return false;
    const data = JSON.parse(stored);
    return data.images && data.images.length > 0;
  } catch {
    return false;
  }
}

// Get captured dataset
export function getCapturedDataset(): TrainingData[] {
  try {
    const stored = localStorage.getItem('asl_captured_dataset');
    if (!stored) return [];
    const data = JSON.parse(stored);
    return data.images.map((img: any) => ({
      letter: img.letter,
      landmarks: img.landmarks
    }));
  } catch {
    return [];
  }
}

// Math utilities for neural network
function relu(x: number): number {
  return Math.max(0, x);
}

function softmax(arr: number[]): number[] {
  const max = Math.max(...arr);
  const exp = arr.map(x => Math.exp(x - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map(x => x / sum);
}

// Normalize landmarks to 63 features
function normalizeLandmarks(landmarks: number[][]): number[] {
  if (landmarks.length !== 21) return [];
  
  // Center around wrist (landmark 0)
  const wrist = landmarks[0].slice();
  const centered = landmarks.map(p => [p[0] - wrist[0], p[1] - wrist[1], p[2] - wrist[2]]);
  
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
  
  return centered.flat();
}

// Simple neural network implementation for browser training
class SimpleNN {
  layers: { weights: number[][]; bias: number[] }[] = [];
  
  constructor(inputSize: number, hiddenSizes: number[], outputSize: number) {
    let prevSize = inputSize;
    
    for (const size of hiddenSizes) {
      this.layers.push({
        weights: this.randomMatrix(prevSize, size),
        bias: new Array(size).fill(0)
      });
      prevSize = size;
    }
    
    // Output layer
    this.layers.push({
      weights: this.randomMatrix(prevSize, outputSize),
      bias: new Array(outputSize).fill(0)
    });
  }
  
  randomMatrix(rows: number, cols: number): number[][] {
    const matrix: number[][] = [];
    const scale = Math.sqrt(2.0 / rows); // He initialization
    for (let i = 0; i < rows; i++) {
      matrix.push([]);
      for (let j = 0; j < cols; j++) {
        matrix[i].push((Math.random() * 2 - 1) * scale);
      }
    }
    return matrix;
  }
  
  forward(input: number[]): { activations: number[][]; preActivations: number[][] } {
    const activations: number[][] = [input];
    const preActivations: number[][] = [];
    
    let x = input;
    for (let l = 0; l < this.layers.length; l++) {
      const layer = this.layers[l];
      const z: number[] = [];
      
      for (let j = 0; j < layer.weights[0].length; j++) {
        let sum = layer.bias[j];
        for (let i = 0; i < x.length; i++) {
          sum += x[i] * layer.weights[i][j];
        }
        z.push(sum);
      }
      
      preActivations.push(z);
      
      // Apply activation (ReLU for hidden, none for output)
      if (l < this.layers.length - 1) {
        x = z.map(relu);
      } else {
        x = softmax(z);
      }
      
      activations.push(x);
    }
    
    return { activations, preActivations };
  }
  
  backward(
    activations: number[][],
    preActivations: number[][],
    target: number[],
    learningRate: number
  ): number {
    const numLayers = this.layers.length;
    const deltas: number[][] = [];
    
    // Output layer delta (softmax + cross-entropy derivative = output - target)
    const output = activations[activations.length - 1];
    const outputDelta = output.map((o, i) => o - target[i]);
    deltas.unshift(outputDelta);
    
    // Loss (cross-entropy)
    let loss = 0;
    for (let i = 0; i < target.length; i++) {
      if (target[i] > 0) {
        loss -= target[i] * Math.log(output[i] + 1e-10);
      }
    }
    
    // Backpropagate through hidden layers
    for (let l = numLayers - 2; l >= 0; l--) {
      const nextDelta = deltas[0];
      const nextWeights = this.layers[l + 1].weights;
      const preAct = preActivations[l];
      
      const delta: number[] = [];
      for (let i = 0; i < this.layers[l].weights[0].length; i++) {
        let sum = 0;
        for (let j = 0; j < nextDelta.length; j++) {
          sum += nextDelta[j] * nextWeights[i][j];
        }
        // ReLU derivative
        delta.push(preAct[i] > 0 ? sum : 0);
      }
      deltas.unshift(delta);
    }
    
    // Update weights
    for (let l = 0; l < numLayers; l++) {
      const delta = deltas[l];
      const input = activations[l];
      
      for (let i = 0; i < this.layers[l].weights.length; i++) {
        for (let j = 0; j < this.layers[l].weights[i].length; j++) {
          this.layers[l].weights[i][j] -= learningRate * delta[j] * input[i];
        }
      }
      
      for (let j = 0; j < this.layers[l].bias.length; j++) {
        this.layers[l].bias[j] -= learningRate * delta[j];
      }
    }
    
    return loss;
  }
  
  predict(input: number[]): number[] {
    const { activations } = this.forward(input);
    return activations[activations.length - 1];
  }
  
  // Export weights in format compatible with nnClassifier
  exportWeights(classes: string[]): any[] {
    const exported: any[] = [];
    
    for (let l = 0; l < this.layers.length; l++) {
      const layer = this.layers[l];
      
      // Dense layer
      exported.push({
        type: 'Dense',
        weights: [layer.weights, layer.bias]
      });
      
      // Add activation for all but last layer
      if (l < this.layers.length - 1) {
        exported.push({
          type: 'Activation',
          activation: 'relu'
        });
      }
    }
    
    return exported;
  }
}

// Main training function
export async function trainModel(
  data: TrainingData[],
  modelName: string,
  epochs: number = 100,
  onProgress?: (progress: TrainingProgress) => void
): Promise<TrainedModel> {
  // Extract unique classes
  const classes = [...new Set(data.map(d => d.letter))].sort();
  
  if (classes.length < 2) {
    throw new Error('Need at least 2 different letters to train');
  }
  
  // Prepare training data
  const X: number[][] = [];
  const Y: number[][] = [];
  
  for (const sample of data) {
    const features = normalizeLandmarks(sample.landmarks);
    if (features.length !== 63) continue;
    
    X.push(features);
    
    // One-hot encode
    const oneHot = new Array(classes.length).fill(0);
    oneHot[classes.indexOf(sample.letter)] = 1;
    Y.push(oneHot);
  }
  
  if (X.length < 10) {
    throw new Error('Need at least 10 valid samples to train');
  }
  
  // Shuffle data
  const indices = Array.from({ length: X.length }, (_, i) => i);
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }
  
  const shuffledX = indices.map(i => X[i]);
  const shuffledY = indices.map(i => Y[i]);
  
  // Split train/test (80/20)
  const splitIdx = Math.floor(shuffledX.length * 0.8);
  const trainX = shuffledX.slice(0, splitIdx);
  const trainY = shuffledY.slice(0, splitIdx);
  const testX = shuffledX.slice(splitIdx);
  const testY = shuffledY.slice(splitIdx);
  
  // Create network
  const nn = new SimpleNN(63, [128, 64, 32], classes.length);
  
  const learningRate = 0.01;
  let bestAccuracy = 0;
  
  // Training loop
  for (let epoch = 0; epoch < epochs; epoch++) {
    let totalLoss = 0;
    
    // Mini-batch training
    for (let i = 0; i < trainX.length; i++) {
      const { activations, preActivations } = nn.forward(trainX[i]);
      totalLoss += nn.backward(activations, preActivations, trainY[i], learningRate);
    }
    
    // Calculate accuracy on test set
    let correct = 0;
    for (let i = 0; i < testX.length; i++) {
      const pred = nn.predict(testX[i]);
      const predIdx = pred.indexOf(Math.max(...pred));
      const trueIdx = testY[i].indexOf(1);
      if (predIdx === trueIdx) correct++;
    }
    
    const accuracy = testX.length > 0 ? correct / testX.length : 0;
    if (accuracy > bestAccuracy) bestAccuracy = accuracy;
    
    const avgLoss = totalLoss / trainX.length;
    
    // Report progress
    if (onProgress) {
      onProgress({
        epoch: epoch + 1,
        totalEpochs: epochs,
        loss: avgLoss,
        accuracy: accuracy,
        status: `Training... ${Math.round((epoch + 1) / epochs * 100)}%`
      });
    }
    
    // Allow UI to update
    if (epoch % 10 === 0) {
      await new Promise(resolve => setTimeout(resolve, 0));
    }
  }
  
  // Export model
  const weights = nn.exportWeights(classes);
  
  const trainedModel: TrainedModel = {
    name: modelName,
    classes,
    weights,
    trainedAt: new Date().toISOString(),
    accuracy: bestAccuracy,
    totalSamples: X.length
  };
  
  // Save to localStorage
  saveCustomModel(trainedModel);
  
  return trainedModel;
}

// Download model as JSON file
export function downloadModelAsJson(model: TrainedModel): void {
  const exportData = {
    name: model.name,
    classes: model.classes,
    weights: model.weights,
    trainedAt: model.trainedAt,
    accuracy: model.accuracy,
    totalSamples: model.totalSamples
  };
  
  const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${model.name}_weights.json`;
  a.click();
  URL.revokeObjectURL(url);
}
