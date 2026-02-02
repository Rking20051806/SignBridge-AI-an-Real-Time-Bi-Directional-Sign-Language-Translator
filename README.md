<div align="center">

# 🤝 SignBridge AI

### Bi-Directional Sign Language Translation System

*Breaking barriers between hearing and deaf communities with AI-powered real-time sign language translation*

[![Live Demo](https://img.shields.io/badge/🔴_Live_Demo-Visit_Site-red?style=for-the-badge&logo=vercel)](https://bidirectional-sign-translator.vercel.app)
[![GitHub](https://img.shields.io/badge/📂_GitHub-Repository-black?style=for-the-badge&logo=github)](https://github.com/Rking18062005/bidirectional-sign-translator)

![React](https://img.shields.io/badge/React-18.3.1-61DAFB?logo=react&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-5.6-3178C6?logo=typescript&logoColor=white)
![Vite](https://img.shields.io/badge/Vite-6.0-646CFF?logo=vite&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3.4-06B6D4?logo=tailwindcss&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Vision-FF6F00?logo=google&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow.js-Neural_Network-FF6F00?logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

## 📋 Table of Contents

- [About](#-about)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Technology Stack](#-technology-stack)
- [Algorithms & Techniques](#-algorithms--techniques)
- [Installation](#-installation)
- [Usage Guide](#-usage-guide)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Limitations](#-limitations)
- [Future Scope](#-future-scope)
- [Contributing](#-contributing)
- [License](#-license)

---

## 📖 About

**SignBridge AI** is an innovative web application that enables **real-time, bi-directional translation** between American Sign Language (ASL) and text/speech. The system leverages cutting-edge AI technologies including computer vision, neural networks, and natural language processing to bridge communication gaps between deaf and hearing communities.

### Key Highlights

- 🎯 **91.96% Accuracy** on ASL alphabet recognition using trained neural network
- ⚡ **Real-time Processing** at 30+ FPS with GPU acceleration
- 🔒 **100% Offline Capable** - No API calls required for sign detection
- 🎨 **Custom Model Training** - Train your own sign models directly in the browser
- 📱 **Fully Responsive** - Optimized for all devices and screen sizes
- 🌐 **Cross-Platform** - Works on Windows, macOS, Linux, Android, iOS

### Platform Compatibility

| Platform | Status | Notes |
|----------|--------|-------|
| **Windows** | ✅ Fully Supported | Chrome, Firefox, Edge |
| **macOS** | ✅ Fully Supported | Chrome, Safari, Firefox |
| **Linux** | ✅ Fully Supported | Chrome, Firefox |
| **Android** | ✅ Fully Supported | Chrome (recommended), Firefox |
| **iOS/iPadOS** | ✅ Fully Supported | Safari (recommended), Chrome |

### Responsive Design Features

- 📲 **Mobile-First Design** - Touch-friendly interface with optimized layouts
- 🖥️ **Desktop Optimized** - Full feature access with expanded UI
- 📱 **Tablet Support** - Adaptive grid layouts for medium screens
- 🔄 **Orientation Aware** - Works in both portrait and landscape modes
- 👆 **Touch Gestures** - Native-like touch interactions on mobile devices
- 🎯 **Safe Area Support** - Proper padding for notched devices (iPhone X+)

---

## ✨ Features

### 1. Sign to Text Translation
- Real-time ASL alphabet recognition (A-Z)
- Live hand tracking with colored landmark visualization
- Confidence scoring for each prediction
- Prediction stabilization to reduce flickering
- Support for webcam and IP camera inputs
- **Model Switching** - Switch between default and custom trained models

### 2. Text to Sign Translation
- Convert text/sentences to ASL fingerspelling
- Animated sign image display
- Adjustable playback speed
- Character-by-character breakdown
- Support for spaces and special handling

### 3. Hearing to Deaf Communication
- Google Gemini AI powered text generation
- Context-aware responses
- Natural language processing

### 4. Dataset Capture & Custom Training
- **Custom Sign Names** - Define any sign (not limited to A-Z)
- **Landmark Extraction** - Automatic hand landmark capture
- **Browser-based Training** - Train neural networks without any server
- **Model Export** - Download trained weights as JSON
- **Model Management** - Save, load, delete, and switch custom models

---

## 🧩 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        SignBridge AI                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│  │   Camera    │───▶│  MediaPipe  │───▶│  Hand Landmarks     │ │
│  │   Input     │    │  HandLandm. │    │  (21 points × 3D)   │ │
│  └─────────────┘    └─────────────┘    └──────────┬──────────┘ │
│                                                    │            │
│                                                    ▼            │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │               Feature Extraction & Normalization            ││
│  │  • Center around wrist (landmark 0)                         ││
│  │  • Scale by palm width                                      ││
│  │  • Extract 63 features (21 × 3 coordinates)                 ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                    │            │
│                                                    ▼            │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Neural Network Classifier                      ││
│  │  Input(63) → Dense(128) → BN → ReLU → Dropout(0.3)         ││
│  │           → Dense(64)  → BN → ReLU → Dropout(0.3)          ││
│  │           → Dense(32)  → BN → ReLU → Dropout(0.2)          ││
│  │           → Dense(26)  → Softmax → Prediction              ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                    │            │
│                                                    ▼            │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Post-Processing & Stabilization                ││
│  │  • Confidence thresholding (>40%)                           ││
│  │  • Majority voting (5-frame buffer)                         ││
│  │  • Result smoothing                                         ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠 Technology Stack

### Frontend Framework
| Technology | Version | Purpose |
|------------|---------|---------|
| **React** | 18.3.1 | UI component library with hooks |
| **TypeScript** | 5.6.2 | Type-safe JavaScript |
| **Vite** | 6.0.5 | Build tool & dev server |
| **TailwindCSS** | 3.4.17 | Utility-first CSS framework |

### Computer Vision & Machine Learning
| Technology | Version | Purpose |
|------------|---------|---------|
| **MediaPipe Tasks Vision** | 0.10.14 | Hand landmark detection (21 3D keypoints) |
| **TensorFlow/Keras** | 2.x | Model training (Python scripts) |
| **Custom Neural Network (JS)** | - | Browser-based inference & training |

### AI & APIs
| Technology | Purpose |
|------------|---------|
| **Google Gemini AI** | Text generation & NLP |
| **localStorage API** | Model persistence in browser |

### Development Tools
| Tool | Purpose |
|------|---------|
| **ESLint** | Code linting |
| **PostCSS** | CSS processing |
| **Autoprefixer** | CSS vendor prefixes |

### Responsive Design Stack
| Technique | Implementation |
|-----------|---------------|
| **Mobile Breakpoints** | `sm:` (640px), `md:` (768px), `lg:` (1024px), `xl:` (1280px) |
| **Viewport Meta** | Full mobile optimization with `viewport-fit=cover` |
| **Safe Area Insets** | `env(safe-area-inset-*)` for notched devices |
| **Touch Targets** | Minimum 44px touch targets on mobile |
| **Flexible Grids** | CSS Grid with responsive column spans |
| **Fluid Typography** | Responsive text sizing with Tailwind classes |

---

## 🧮 Algorithms & Techniques

### 1. Hand Landmark Detection (MediaPipe)

MediaPipe's HandLandmarker uses a **two-stage neural network pipeline**:

```
Stage 1: Palm Detection
├── BlazePalm detector (single-shot detector)
├── Detects palm bounding box
└── ~2.5ms inference time

Stage 2: Hand Landmark Regression
├── Takes cropped palm image
├── Predicts 21 3D keypoints
├── Runs at 30+ FPS on GPU
└── Float16 precision for efficiency
```

**21 Hand Landmarks:**
```
        8   12  16  20      (Fingertips)
        |   |   |   |
    4   7   11  15  19      (DIP joints)
    |   |   |   |   |
    3   6   10  14  18      (PIP joints)
    |   |   |   |   |
    2   5   9   13  17      (MCP joints)
    |    \   \   \  /
    1      \_______/        (CMC/Palm)
    |          |
    0 (WRIST)--+
```

### 2. Feature Normalization Algorithm

```python
# Step 1: Centering around wrist
centered = landmarks - landmarks[0]  # Subtract wrist position

# Step 2: Scale normalization  
palm_scale = distance(landmarks[9], landmarks[0])  # Wrist to middle MCP
normalized = centered / palm_scale

# Output: 63 features (21 landmarks × 3 coordinates)
features = normalized.flatten()  # [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20]
```

**Why Normalize?**
- **Translation Invariance**: Hand position in frame doesn't affect prediction
- **Scale Invariance**: Hand size/distance from camera doesn't matter
- **Consistent Input Range**: Improves neural network convergence

### 3. Neural Network Architecture

```
Layer (type)                Output Shape      Param #   
═══════════════════════════════════════════════════════
Input                       (None, 63)        0         
Dense                       (None, 128)       8,192     
BatchNormalization          (None, 128)       512       
Activation (ReLU)           (None, 128)       0         
Dropout (0.3)               (None, 128)       0         
Dense                       (None, 64)        8,256     
BatchNormalization          (None, 64)        256       
Activation (ReLU)           (None, 64)        0         
Dropout (0.3)               (None, 64)        0         
Dense                       (None, 32)        2,080     
BatchNormalization          (None, 32)        128       
Activation (ReLU)           (None, 32)        0         
Dropout (0.2)               (None, 32)        0         
Dense                       (None, 26)        858       
Activation (Softmax)        (None, 26)        0         
═══════════════════════════════════════════════════════
Total params: 20,282
Trainable params: 19,834
```

### 4. Training Techniques

| Technique | Purpose |
|-----------|---------|
| **Adam Optimizer** | Adaptive learning rate optimization |
| **Batch Normalization** | Faster convergence, internal covariate shift reduction |
| **Dropout** | Prevent overfitting by randomly dropping neurons |
| **He Initialization** | Proper weight initialization for ReLU activations |
| **Categorical Cross-Entropy** | Multi-class classification loss function |
| **Early Stopping** | Prevent overfitting by monitoring validation loss |

### 5. Prediction Stabilization Algorithm

```javascript
// Majority voting with sliding window
const BUFFER_SIZE = 5;
const buffer = []; // Last 5 predictions

function stabilizePrediction(prediction) {
    buffer.push(prediction);
    if (buffer.length > BUFFER_SIZE) buffer.shift();
    
    // Count occurrences
    const counts = {};
    buffer.forEach(p => counts[p] = (counts[p] || 0) + 1);
    
    // Return most common prediction
    return Object.entries(counts)
        .sort((a, b) => b[1] - a[1])[0][0];
}
```

### 6. Browser-Based Training (Custom Models)

```javascript
// Simple Neural Network with Backpropagation
class SimpleNN {
    constructor(layerSizes) {
        // He initialization for weights
        this.weights = [];
        for (let i = 0; i < layerSizes.length - 1; i++) {
            const scale = Math.sqrt(2 / layerSizes[i]);
            this.weights.push(randomMatrix(layerSizes[i], layerSizes[i+1], scale));
        }
    }
    
    forward(input) {
        let x = input;
        for (let i = 0; i < this.weights.length; i++) {
            x = matmul(x, this.weights[i]);
            x = addBias(x, this.biases[i]);
            x = (i === this.weights.length - 1) ? softmax(x) : relu(x);
        }
        return x;
    }
    
    backward(target, learningRate) {
        // Backpropagation: compute gradients via chain rule
        // Update: weights = weights - learningRate * gradients
    }
}
```

---

## 📦 Installation

### Prerequisites

- **Node.js** 18.x or higher
- **npm** 9.x or higher
- Modern browser with WebGL support (Chrome, Firefox, Edge)
- Webcam (for sign detection)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/Rking18062005/bidirectional-sign-translator.git
cd bidirectional-sign-translator

# 2. Install dependencies
npm install

# 3. Set up environment variables (optional - for Gemini AI)
# Create .env file and add your API key:
# VITE_GEMINI_API_KEY=your_api_key_here

# 4. Start development server
npm run dev

# 5. Open in browser
# http://localhost:3000
```

### Build for Production

```bash
# Create optimized build
npm run build

# Preview production build
npm run preview
```

### Deployment on Vercel

SignBridge AI is deployed and hosted on **Vercel** for global accessibility:

```bash
# Install Vercel CLI
npm install -g vercel

# Deploy to Vercel
vercel

# Or deploy directly from GitHub
# 1. Connect your GitHub repo to Vercel
# 2. Vercel auto-detects Vite and configures build
# 3. Push to main branch triggers automatic deployment
```

**Live Demo**: [bidirectional-sign-translator.vercel.app](https://bidirectional-sign-translator.vercel.app)

### Accessing on Different Devices

| Device | Access Method |
|--------|---------------|
| **Desktop** | Open the URL in any modern browser |
| **Mobile (Android)** | Open in Chrome/Firefox, tap "Add to Home Screen" for app-like experience |
| **Mobile (iOS)** | Open in Safari, tap Share → "Add to Home Screen" |
| **Tablet** | Works in both portrait and landscape orientations |

---

## 📖 Usage Guide

### Sign to Text
1. Navigate to **"Sign → Text"** tab
2. Click **"Activate System"** to initialize camera & model
3. Click **"Start Translation"** to begin recognition
4. Show ASL hand signs to the camera
5. View detected letters and confidence scores
6. Access **Settings** to switch between models

### Text to Sign
1. Navigate to **"Text → Sign"** tab
2. Enter text in the input field
3. Click **"Translate"** or press Enter
4. Watch the animated fingerspelling display
5. Adjust speed with the slider

### Custom Model Training
1. Navigate to **"Dataset Capture"** tab
2. Enter a sign name (e.g., "HELLO", "YES", "A")
3. Click **"Add Sign"** to add to your list
4. Start camera and capture 50+ images per sign
5. Expand **"Train Model"** section in sidebar
6. Enter a model name and click **"Train Model"**
7. Wait for training to complete (~30 seconds)
8. Use your model in Sign to Text via **Settings → Model Selection**

---

## 📁 Project Structure

```
signbridge/
├── public/
│   ├── asl_trained_weights.json   # Pre-trained NN weights
│   └── reference/                  # ASL reference images (A-Z)
│
├── components/
│   ├── App.tsx                     # Main app with tab navigation
│   ├── SignToText.tsx              # Sign language recognition
│   ├── TextToSign.tsx              # Text to fingerspelling
│   ├── HearingToDeaf.tsx           # AI-powered communication
│   └── DatasetCapture.tsx          # Dataset capture & training
│
├── services/
│   ├── nnClassifier.ts             # Neural network inference
│   ├── aslClassifier.ts            # Geometric pattern matching (legacy)
│   ├── modelTrainer.ts             # Browser-based training
│   └── geminiService.ts            # Gemini AI integration
│
├── training/                       # Python training scripts
│   ├── train_landmark_model.py     # Main training script
│   ├── export_weights.py           # Export to JSON
│   ├── capture_images.py           # Dataset capture utility
│   └── README.md                   # Training documentation
│
├── constants.ts                    # ASL alphabet mappings
├── types.ts                        # TypeScript interfaces
├── index.tsx                       # Entry point
├── vite.config.ts                  # Vite configuration
└── package.json                    # Dependencies
```

---

## ⚙️ How It Works

### Sign Detection Pipeline

```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Camera Frame │────▶│ MediaPipe        │────▶│ Hand Detected?  │
│  (30 FPS)    │     │ HandLandmarker   │     │                 │
└──────────────┘     └──────────────────┘     └────────┬────────┘
                                                       │
                     ┌─────────────────────────────────┘
                     │ Yes                    No │
                     ▼                           │
         ┌──────────────────────┐               │
         │ Extract 21 Landmarks │               │
         │ (63 features total)  │               │
         └──────────┬───────────┘               │
                    │                           │
                    ▼                           │
         ┌──────────────────────┐               │
         │ Normalize Features   │               │
         │ (center + scale)     │               │
         └──────────┬───────────┘               │
                    │                           │
                    ▼                           │
         ┌──────────────────────┐               │
         │ Neural Network       │               │
         │ Forward Pass         │               │
         └──────────┬───────────┘               │
                    │                           │
                    ▼                           │
         ┌──────────────────────┐               │
         │ Confidence > 40%?    │───No─────────┤
         └──────────┬───────────┘               │
                    │ Yes                       │
                    ▼                           │
         ┌──────────────────────┐               │
         │ Stabilization Buffer │               │
         │ (Majority Vote)      │               │
         └──────────┬───────────┘               │
                    │                           │
                    ▼                           ▼
         ┌──────────────────────┐    ┌─────────────────┐
         │ Display Prediction   │    │ No Hand Message │
         │ with Confidence      │    │                 │
         └──────────────────────┘    └─────────────────┘
```

---

## ⚠️ Limitations

### Current Limitations

| Limitation | Description |
|------------|-------------|
| **Static Signs Only** | Only recognizes static hand poses, not dynamic gestures |
| **Single Hand** | Processes one hand at a time (dominant hand) |
| **ASL Alphabet** | Default model limited to fingerspelling (A-Z) |
| **Lighting Sensitivity** | Performance degrades in poor lighting conditions |
| **Background Noise** | Complex backgrounds can affect detection |
| **Motion Blur** | Fast movements cause detection failures |
| **Browser Memory** | Large datasets may cause memory issues |
| **Training Time** | Browser-based training is slower than GPU training |

### Known Issues

1. **Letters J & Z**: Require motion; current system detects static approximations
2. **Similar Signs**: Letters like M/N, A/S/E can be confused
3. **Webcam Quality**: Low-resolution cameras reduce accuracy
4. **Mobile Performance**: Slower inference on mobile devices
5. **No Word Recognition**: Only individual letters/signs, no word-level detection

---

## 🚀 Future Scope

### Short-Term Improvements (v2.0)

| Feature | Description | Priority |
|---------|-------------|----------|
| **Dynamic Gesture Recognition** | Support for J, Z, and motion-based signs | High |
| **Word-Level Recognition** | Recognize common ASL words, not just letters | High |
| **Two-Hand Support** | Process both hands simultaneously | Medium |
| **Voice Input** | Speech-to-text then translate to sign | Medium |
| **Offline PWA** | Full offline support as Progressive Web App | Medium |

### Medium-Term Enhancements (v3.0)

| Feature | Description |
|---------|-------------|
| **Sentence Translation** | Full sentence ASL to English translation |
| **LSTM/Transformer Models** | Temporal models for gesture sequences |
| **Multi-Language Support** | BSL (British), ISL (Indian), JSL (Japanese), etc. |
| **AR Overlay** | Augmented reality sign overlays |
| **Mobile App** | Native iOS/Android applications |

### Long-Term Vision (v4.0+)

| Feature | Description |
|---------|-------------|
| **Full Conversation Mode** | Real-time bi-directional video calls |
| **Sign Language Avatar** | 3D avatar performing signs |
| **Cloud Model Training** | Upload datasets, train on cloud GPUs |
| **API Service** | REST/WebSocket API for third-party integration |
| **Edge Deployment** | Run on Raspberry Pi, NVIDIA Jetson |

### Potential Integrations

- Video conferencing (Zoom, Teams, Google Meet)
- Education platforms (LMS, online courses)
- Healthcare systems (hospital kiosks, telehealth)
- Smart glasses and wearable devices
- IoT devices and smart home systems

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Development Guidelines

- Follow TypeScript best practices
- Write meaningful commit messages
- Add tests for new features
- Update documentation as needed

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Model Accuracy (Test Set) | 91.96% |
| Inference Time | ~5ms per frame |
| Detection FPS | 30+ FPS |
| Model Size (JSON) | ~80KB |
| Bundle Size (Gzipped) | ~120KB |

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **MediaPipe Team** - Excellent hand tracking models
- **TensorFlow/Keras** - Training framework
- **Google Gemini** - AI-powered text generation
- **ASL Community** - Sign language resources and references
- **Open Source Contributors** - Various libraries and tools

---

<div align="center">

**Made with ❤️ for Accessibility**

*Bridging the communication gap, one sign at a time*

[![Star on GitHub](https://img.shields.io/github/stars/Rking18062005/bidirectional-sign-translator?style=social)](https://github.com/Rking18062005/bidirectional-sign-translator)

</div>
