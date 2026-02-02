/**
 * Advanced Offline ASL Alphabet Classifier using MediaPipe Hand Landmarks
 * Uses geometric pattern matching based on ASL letter specifications
 * No API required - fully offline
 */

// MediaPipe Hand Landmark indices
const WRIST = 0;
const THUMB_CMC = 1, THUMB_MCP = 2, THUMB_IP = 3, THUMB_TIP = 4;
const INDEX_MCP = 5, INDEX_PIP = 6, INDEX_DIP = 7, INDEX_TIP = 8;
const MIDDLE_MCP = 9, MIDDLE_PIP = 10, MIDDLE_DIP = 11, MIDDLE_TIP = 12;
const RING_MCP = 13, RING_PIP = 14, RING_DIP = 15, RING_TIP = 16;
const PINKY_MCP = 17, PINKY_PIP = 18, PINKY_DIP = 19, PINKY_TIP = 20;

interface Landmark {
  x: number;
  y: number;
  z: number;
}

interface HandFeatures {
  // Finger extension states (0-1, 1 = fully extended)
  thumbExtension: number;
  indexExtension: number;
  middleExtension: number;
  ringExtension: number;
  pinkyExtension: number;
  
  // Finger curl states (0-1, 1 = fully curled)
  thumbCurl: number;
  indexCurl: number;
  middleCurl: number;
  ringCurl: number;
  pinkyCurl: number;
  
  // Finger spread (distance between adjacent fingertips)
  thumbIndexSpread: number;
  indexMiddleSpread: number;
  middleRingSpread: number;
  ringPinkySpread: number;
  
  // Special features
  thumbIndexTouch: boolean;
  thumbMiddleTouch: boolean;
  thumbRingTouch: boolean;
  thumbPinkyTouch: boolean;
  indexMiddleTouch: boolean;
  
  // Hand orientation
  palmFacing: 'camera' | 'away' | 'side';
  fingersPointing: 'up' | 'down' | 'side' | 'forward';
  
  // Thumb position relative to palm
  thumbPosition: 'out' | 'across' | 'tucked' | 'up';
}

interface LetterTemplate {
  letter: string;
  match: (features: HandFeatures) => number; // Returns confidence 0-1
}

// ============ UTILITY FUNCTIONS ============

function distance(a: Landmark, b: Landmark): number {
  return Math.sqrt(
    Math.pow(a.x - b.x, 2) + 
    Math.pow(a.y - b.y, 2) + 
    Math.pow(a.z - b.z, 2)
  );
}

function distance2D(a: Landmark, b: Landmark): number {
  return Math.sqrt(
    Math.pow(a.x - b.x, 2) + 
    Math.pow(a.y - b.y, 2)
  );
}

// Normalize landmarks to be scale-invariant (palm width = 1)
function normalizeLandmarks(landmarks: Landmark[]): Landmark[] {
  const palmWidth = distance(landmarks[INDEX_MCP], landmarks[PINKY_MCP]);
  if (palmWidth === 0) return landmarks;
  
  const wrist = landmarks[WRIST];
  return landmarks.map(l => ({
    x: (l.x - wrist.x) / palmWidth,
    y: (l.y - wrist.y) / palmWidth,
    z: (l.z - wrist.z) / palmWidth
  }));
}

// Calculate finger extension (0 = curled, 1 = extended)
function getFingerExtension(
  landmarks: Landmark[], 
  mcpIdx: number, 
  pipIdx: number, 
  tipIdx: number
): number {
  const mcp = landmarks[mcpIdx];
  const pip = landmarks[pipIdx];
  const tip = landmarks[tipIdx];
  const wrist = landmarks[WRIST];
  
  // Distance from tip to wrist vs mcp to wrist
  const tipToWrist = distance(tip, wrist);
  const mcpToWrist = distance(mcp, wrist);
  
  // Extension ratio
  const ratio = tipToWrist / (mcpToWrist + 0.001);
  
  // Normalize to 0-1 range (typically 1.0-2.5 for finger extension)
  return Math.min(1, Math.max(0, (ratio - 0.8) / 1.2));
}

// Calculate finger curl (0 = straight, 1 = fully curled)
function getFingerCurl(
  landmarks: Landmark[], 
  mcpIdx: number, 
  pipIdx: number, 
  dipIdx: number,
  tipIdx: number
): number {
  const mcp = landmarks[mcpIdx];
  const pip = landmarks[pipIdx];
  const tip = landmarks[tipIdx];
  
  // Vector from mcp to pip
  const v1 = { x: pip.x - mcp.x, y: pip.y - mcp.y };
  // Vector from pip to tip
  const v2 = { x: tip.x - pip.x, y: tip.y - pip.y };
  
  // Dot product for angle
  const dot = v1.x * v2.x + v1.y * v2.y;
  const mag1 = Math.sqrt(v1.x * v1.x + v1.y * v1.y);
  const mag2 = Math.sqrt(v2.x * v2.x + v2.y * v2.y);
  
  if (mag1 === 0 || mag2 === 0) return 0;
  
  const cosAngle = dot / (mag1 * mag2);
  // Convert to curl (1 when angle is 180° = straight, 0 when bent)
  const curl = 1 - (cosAngle + 1) / 2;
  
  return Math.min(1, Math.max(0, curl * 2)); // Scale up
}

// Get thumb extension (different geometry)
function getThumbExtension(landmarks: Landmark[]): number {
  const thumbTip = landmarks[THUMB_TIP];
  const thumbMcp = landmarks[THUMB_MCP];
  const indexMcp = landmarks[INDEX_MCP];
  
  // Thumb is extended if tip is far from index MCP
  const tipToIndex = distance(thumbTip, indexMcp);
  const mcpToIndex = distance(thumbMcp, indexMcp);
  
  return Math.min(1, tipToIndex / (mcpToIndex + 0.001) - 0.5);
}

// Check palm orientation
function getPalmFacing(landmarks: Landmark[]): 'camera' | 'away' | 'side' {
  // Use z-coordinates of palm landmarks
  const palmZ = (landmarks[INDEX_MCP].z + landmarks[PINKY_MCP].z + landmarks[WRIST].z) / 3;
  const knuckleZ = (landmarks[INDEX_PIP].z + landmarks[MIDDLE_PIP].z) / 2;
  
  const zDiff = knuckleZ - palmZ;
  
  if (Math.abs(zDiff) < 0.02) return 'side';
  return zDiff > 0 ? 'camera' : 'away';
}

// Check finger pointing direction
function getFingersPointing(landmarks: Landmark[]): 'up' | 'down' | 'side' | 'forward' {
  const wrist = landmarks[WRIST];
  const middleTip = landmarks[MIDDLE_TIP];
  
  const dx = middleTip.x - wrist.x;
  const dy = middleTip.y - wrist.y;
  const dz = middleTip.z - wrist.z;
  
  // Check if pointing forward/backward (z-axis dominant)
  if (Math.abs(dz) > Math.abs(dx) && Math.abs(dz) > Math.abs(dy)) {
    return 'forward';
  }
  
  // Check horizontal vs vertical
  if (Math.abs(dx) > Math.abs(dy)) {
    return 'side';
  }
  
  return dy < 0 ? 'up' : 'down';
}

// Get thumb position
function getThumbPosition(landmarks: Landmark[]): 'out' | 'across' | 'tucked' | 'up' {
  const thumbTip = landmarks[THUMB_TIP];
  const indexMcp = landmarks[INDEX_MCP];
  const middleMcp = landmarks[MIDDLE_MCP];
  const pinkyMcp = landmarks[PINKY_MCP];
  const wrist = landmarks[WRIST];
  
  const thumbToIndex = distance(thumbTip, indexMcp);
  const thumbToMiddle = distance(thumbTip, middleMcp);
  const palmWidth = distance(indexMcp, pinkyMcp);
  
  // Thumb pointing up
  if (thumbTip.y < wrist.y - palmWidth * 0.3) {
    return 'up';
  }
  
  // Thumb across palm (near middle or beyond)
  if (thumbToMiddle < palmWidth * 0.5) {
    return 'across';
  }
  
  // Thumb tucked (below fingers)
  if (thumbTip.y > indexMcp.y && thumbToIndex < palmWidth * 0.3) {
    return 'tucked';
  }
  
  return 'out';
}

// Extract all features from landmarks
function extractFeatures(landmarks: Landmark[]): HandFeatures {
  const norm = normalizeLandmarks(landmarks);
  
  // Finger extensions
  const thumbExtension = getThumbExtension(norm);
  const indexExtension = getFingerExtension(norm, INDEX_MCP, INDEX_PIP, INDEX_TIP);
  const middleExtension = getFingerExtension(norm, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP);
  const ringExtension = getFingerExtension(norm, RING_MCP, RING_PIP, RING_TIP);
  const pinkyExtension = getFingerExtension(norm, PINKY_MCP, PINKY_PIP, PINKY_TIP);
  
  // Finger curls
  const thumbCurl = getFingerCurl(norm, THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP);
  const indexCurl = getFingerCurl(norm, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP);
  const middleCurl = getFingerCurl(norm, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP);
  const ringCurl = getFingerCurl(norm, RING_MCP, RING_PIP, RING_DIP, RING_TIP);
  const pinkyCurl = getFingerCurl(norm, PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP);
  
  // Finger spreads (normalized)
  const palmWidth = distance(norm[INDEX_MCP], norm[PINKY_MCP]);
  const thumbIndexSpread = distance(norm[THUMB_TIP], norm[INDEX_TIP]) / (palmWidth + 0.001);
  const indexMiddleSpread = distance(norm[INDEX_TIP], norm[MIDDLE_TIP]) / (palmWidth + 0.001);
  const middleRingSpread = distance(norm[MIDDLE_TIP], norm[RING_TIP]) / (palmWidth + 0.001);
  const ringPinkySpread = distance(norm[RING_TIP], norm[PINKY_TIP]) / (palmWidth + 0.001);
  
  // Touch detection (threshold based)
  const touchThreshold = 0.15;
  const thumbIndexTouch = distance(norm[THUMB_TIP], norm[INDEX_TIP]) < touchThreshold;
  const thumbMiddleTouch = distance(norm[THUMB_TIP], norm[MIDDLE_TIP]) < touchThreshold;
  const thumbRingTouch = distance(norm[THUMB_TIP], norm[RING_TIP]) < touchThreshold;
  const thumbPinkyTouch = distance(norm[THUMB_TIP], norm[PINKY_TIP]) < touchThreshold;
  const indexMiddleTouch = distance(norm[INDEX_TIP], norm[MIDDLE_TIP]) < touchThreshold;
  
  return {
    thumbExtension,
    indexExtension,
    middleExtension,
    ringExtension,
    pinkyExtension,
    thumbCurl,
    indexCurl,
    middleCurl,
    ringCurl,
    pinkyCurl,
    thumbIndexSpread,
    indexMiddleSpread,
    middleRingSpread,
    ringPinkySpread,
    thumbIndexTouch,
    thumbMiddleTouch,
    thumbRingTouch,
    thumbPinkyTouch,
    indexMiddleTouch,
    palmFacing: getPalmFacing(landmarks),
    fingersPointing: getFingersPointing(landmarks),
    thumbPosition: getThumbPosition(landmarks)
  };
}

// ============ ASL LETTER TEMPLATES ============
// Each template defines expected features for an ASL letter

const ASL_TEMPLATES: LetterTemplate[] = [
  // A - Fist with thumb on side
  {
    letter: 'A',
    match: (f) => {
      let score = 0;
      // All fingers curled
      if (f.indexCurl > 0.5) score += 0.2;
      if (f.middleCurl > 0.5) score += 0.2;
      if (f.ringCurl > 0.5) score += 0.2;
      if (f.pinkyCurl > 0.5) score += 0.2;
      // Thumb out to side (not across)
      if (f.thumbPosition === 'out' || f.thumbPosition === 'up') score += 0.2;
      return score;
    }
  },
  
  // B - Flat hand, fingers together, thumb tucked
  {
    letter: 'B',
    match: (f) => {
      let score = 0;
      // All 4 fingers extended
      if (f.indexExtension > 0.6) score += 0.15;
      if (f.middleExtension > 0.6) score += 0.15;
      if (f.ringExtension > 0.6) score += 0.15;
      if (f.pinkyExtension > 0.6) score += 0.15;
      // Fingers together (not spread)
      if (f.indexMiddleSpread < 0.4) score += 0.1;
      if (f.middleRingSpread < 0.4) score += 0.1;
      // Thumb tucked or across
      if (f.thumbPosition === 'tucked' || f.thumbPosition === 'across') score += 0.2;
      return score;
    }
  },
  
  // C - Curved hand forming C shape
  {
    letter: 'C',
    match: (f) => {
      let score = 0;
      // Fingers partially curled (C shape)
      if (f.indexCurl > 0.2 && f.indexCurl < 0.7) score += 0.2;
      if (f.middleCurl > 0.2 && f.middleCurl < 0.7) score += 0.15;
      if (f.ringCurl > 0.2 && f.ringCurl < 0.7) score += 0.15;
      // Thumb out forming C
      if (f.thumbIndexSpread > 0.3 && f.thumbIndexSpread < 0.8) score += 0.3;
      if (f.thumbPosition === 'out') score += 0.2;
      return score;
    }
  },
  
  // D - Index up, thumb touches middle
  {
    letter: 'D',
    match: (f) => {
      let score = 0;
      // Index extended
      if (f.indexExtension > 0.6) score += 0.3;
      // Other fingers curled
      if (f.middleCurl > 0.4) score += 0.15;
      if (f.ringCurl > 0.4) score += 0.15;
      if (f.pinkyCurl > 0.4) score += 0.1;
      // Thumb touches middle finger or forms circle with curled fingers
      if (f.thumbMiddleTouch || f.thumbPosition === 'across') score += 0.3;
      return score;
    }
  },
  
  // E - All fingers curled, thumb across
  {
    letter: 'E',
    match: (f) => {
      let score = 0;
      // All fingers curled
      if (f.indexCurl > 0.6) score += 0.2;
      if (f.middleCurl > 0.6) score += 0.15;
      if (f.ringCurl > 0.6) score += 0.15;
      if (f.pinkyCurl > 0.6) score += 0.15;
      // Thumb across palm
      if (f.thumbPosition === 'across' || f.thumbPosition === 'tucked') score += 0.35;
      return score;
    }
  },
  
  // F - Thumb and index circle, other 3 extended
  {
    letter: 'F',
    match: (f) => {
      let score = 0;
      // Thumb and index touching
      if (f.thumbIndexTouch) score += 0.35;
      // Middle, ring, pinky extended
      if (f.middleExtension > 0.5) score += 0.2;
      if (f.ringExtension > 0.5) score += 0.15;
      if (f.pinkyExtension > 0.5) score += 0.15;
      // Index partially curled for circle
      if (f.indexCurl > 0.2) score += 0.15;
      return score;
    }
  },
  
  // G - Index and thumb pointing sideways
  {
    letter: 'G',
    match: (f) => {
      let score = 0;
      // Index extended
      if (f.indexExtension > 0.5) score += 0.25;
      // Thumb extended
      if (f.thumbExtension > 0.3) score += 0.2;
      // Other fingers curled
      if (f.middleCurl > 0.4) score += 0.1;
      if (f.ringCurl > 0.4) score += 0.1;
      if (f.pinkyCurl > 0.4) score += 0.1;
      // Pointing sideways
      if (f.fingersPointing === 'side') score += 0.25;
      return score;
    }
  },
  
  // H - Index and middle extended sideways
  {
    letter: 'H',
    match: (f) => {
      let score = 0;
      // Index and middle extended
      if (f.indexExtension > 0.5) score += 0.2;
      if (f.middleExtension > 0.5) score += 0.2;
      // Ring and pinky curled
      if (f.ringCurl > 0.4) score += 0.15;
      if (f.pinkyCurl > 0.4) score += 0.15;
      // Pointing sideways
      if (f.fingersPointing === 'side') score += 0.3;
      return score;
    }
  },
  
  // I - Only pinky extended
  {
    letter: 'I',
    match: (f) => {
      let score = 0;
      // Only pinky extended
      if (f.pinkyExtension > 0.5) score += 0.35;
      // Other fingers curled
      if (f.indexCurl > 0.5) score += 0.15;
      if (f.middleCurl > 0.5) score += 0.15;
      if (f.ringCurl > 0.5) score += 0.15;
      // Thumb tucked
      if (f.thumbPosition !== 'out') score += 0.2;
      return score;
    }
  },
  
  // J - Like I but with movement (static = I shape pointing down)
  {
    letter: 'J',
    match: (f) => {
      let score = 0;
      // Same as I but pointing down/side for J motion
      if (f.pinkyExtension > 0.5) score += 0.3;
      if (f.indexCurl > 0.5) score += 0.15;
      if (f.middleCurl > 0.5) score += 0.15;
      if (f.ringCurl > 0.5) score += 0.15;
      if (f.fingersPointing === 'down' || f.fingersPointing === 'side') score += 0.25;
      return score;
    }
  },
  
  // K - Index and middle up in V, thumb up between
  {
    letter: 'K',
    match: (f) => {
      let score = 0;
      // Index and middle extended
      if (f.indexExtension > 0.5) score += 0.2;
      if (f.middleExtension > 0.5) score += 0.2;
      // Spread apart
      if (f.indexMiddleSpread > 0.3) score += 0.15;
      // Ring and pinky curled
      if (f.ringCurl > 0.4) score += 0.1;
      if (f.pinkyCurl > 0.4) score += 0.1;
      // Thumb up/out
      if (f.thumbPosition === 'up' || f.thumbExtension > 0.4) score += 0.25;
      return score;
    }
  },
  
  // L - Index and thumb in L shape
  {
    letter: 'L',
    match: (f) => {
      let score = 0;
      // Index extended
      if (f.indexExtension > 0.6) score += 0.25;
      // Thumb extended out
      if (f.thumbExtension > 0.4) score += 0.25;
      // Large spread between thumb and index
      if (f.thumbIndexSpread > 0.5) score += 0.2;
      // Other fingers curled
      if (f.middleCurl > 0.4) score += 0.1;
      if (f.ringCurl > 0.4) score += 0.1;
      if (f.pinkyCurl > 0.4) score += 0.1;
      return score;
    }
  },
  
  // M - Fist with thumb under 3 fingers
  {
    letter: 'M',
    match: (f) => {
      let score = 0;
      // All fingers curled
      if (f.indexCurl > 0.5) score += 0.15;
      if (f.middleCurl > 0.5) score += 0.15;
      if (f.ringCurl > 0.5) score += 0.15;
      if (f.pinkyCurl > 0.5) score += 0.15;
      // Thumb tucked under
      if (f.thumbPosition === 'tucked' || f.thumbPosition === 'across') score += 0.4;
      return score;
    }
  },
  
  // N - Fist with thumb under 2 fingers
  {
    letter: 'N',
    match: (f) => {
      let score = 0;
      // All fingers curled
      if (f.indexCurl > 0.5) score += 0.15;
      if (f.middleCurl > 0.5) score += 0.15;
      if (f.ringCurl > 0.5) score += 0.15;
      if (f.pinkyCurl > 0.5) score += 0.15;
      // Thumb across (similar to M but position differs)
      if (f.thumbPosition === 'across') score += 0.4;
      return score;
    }
  },
  
  // O - All fingertips touching thumb (circle)
  {
    letter: 'O',
    match: (f) => {
      let score = 0;
      // Thumb-index touching or close
      if (f.thumbIndexTouch) score += 0.3;
      else if (f.thumbIndexSpread < 0.3) score += 0.15;
      // Fingers curved inward
      if (f.indexCurl > 0.3 && f.indexCurl < 0.8) score += 0.15;
      if (f.middleCurl > 0.3 && f.middleCurl < 0.8) score += 0.15;
      if (f.ringCurl > 0.3) score += 0.1;
      if (f.pinkyCurl > 0.3) score += 0.1;
      // Fingers together
      if (f.indexMiddleSpread < 0.3) score += 0.1;
      if (f.middleRingSpread < 0.3) score += 0.1;
      return score;
    }
  },
  
  // P - Like K but pointing down
  {
    letter: 'P',
    match: (f) => {
      let score = 0;
      // Index and middle extended
      if (f.indexExtension > 0.4) score += 0.2;
      if (f.middleExtension > 0.4) score += 0.2;
      // Pointing down
      if (f.fingersPointing === 'down') score += 0.35;
      // Ring and pinky curled
      if (f.ringCurl > 0.4) score += 0.1;
      if (f.pinkyCurl > 0.4) score += 0.1;
      return score;
    }
  },
  
  // Q - Thumb and index pointing down
  {
    letter: 'Q',
    match: (f) => {
      let score = 0;
      // Index extended or partially
      if (f.indexExtension > 0.3) score += 0.25;
      // Thumb extended
      if (f.thumbExtension > 0.3) score += 0.2;
      // Pointing down
      if (f.fingersPointing === 'down') score += 0.35;
      // Other fingers curled
      if (f.middleCurl > 0.4) score += 0.1;
      if (f.ringCurl > 0.4) score += 0.1;
      return score;
    }
  },
  
  // R - Index and middle crossed
  {
    letter: 'R',
    match: (f) => {
      let score = 0;
      // Index and middle extended
      if (f.indexExtension > 0.5) score += 0.2;
      if (f.middleExtension > 0.5) score += 0.2;
      // Very close together (crossed)
      if (f.indexMiddleTouch || f.indexMiddleSpread < 0.15) score += 0.35;
      // Other curled
      if (f.ringCurl > 0.4) score += 0.1;
      if (f.pinkyCurl > 0.4) score += 0.1;
      return score;
    }
  },
  
  // S - Fist with thumb across fingers
  {
    letter: 'S',
    match: (f) => {
      let score = 0;
      // All fingers tightly curled
      if (f.indexCurl > 0.6) score += 0.15;
      if (f.middleCurl > 0.6) score += 0.15;
      if (f.ringCurl > 0.6) score += 0.15;
      if (f.pinkyCurl > 0.6) score += 0.15;
      // Thumb across (not sticking up like A)
      if (f.thumbPosition === 'across') score += 0.3;
      if (f.thumbExtension < 0.3) score += 0.1;
      return score;
    }
  },
  
  // T - Fist with thumb between index and middle
  {
    letter: 'T',
    match: (f) => {
      let score = 0;
      // Fingers curled
      if (f.indexCurl > 0.5) score += 0.15;
      if (f.middleCurl > 0.5) score += 0.15;
      if (f.ringCurl > 0.5) score += 0.15;
      if (f.pinkyCurl > 0.5) score += 0.15;
      // Thumb tucked between index and middle
      if (f.thumbPosition === 'tucked' || f.thumbPosition === 'across') score += 0.25;
      if (f.thumbIndexTouch || f.thumbMiddleTouch) score += 0.15;
      return score;
    }
  },
  
  // U - Index and middle extended together
  {
    letter: 'U',
    match: (f) => {
      let score = 0;
      // Index and middle extended
      if (f.indexExtension > 0.6) score += 0.2;
      if (f.middleExtension > 0.6) score += 0.2;
      // Close together
      if (f.indexMiddleSpread < 0.25) score += 0.25;
      // Other curled
      if (f.ringCurl > 0.4) score += 0.1;
      if (f.pinkyCurl > 0.4) score += 0.1;
      // Pointing up
      if (f.fingersPointing === 'up') score += 0.15;
      return score;
    }
  },
  
  // V - Index and middle spread (peace sign)
  {
    letter: 'V',
    match: (f) => {
      let score = 0;
      // Index and middle extended
      if (f.indexExtension > 0.6) score += 0.2;
      if (f.middleExtension > 0.6) score += 0.2;
      // Spread apart
      if (f.indexMiddleSpread > 0.25) score += 0.25;
      // Other curled
      if (f.ringCurl > 0.4) score += 0.1;
      if (f.pinkyCurl > 0.4) score += 0.1;
      // Pointing up
      if (f.fingersPointing === 'up') score += 0.15;
      return score;
    }
  },
  
  // W - Index, middle, ring extended and spread
  {
    letter: 'W',
    match: (f) => {
      let score = 0;
      // Three fingers extended
      if (f.indexExtension > 0.5) score += 0.2;
      if (f.middleExtension > 0.5) score += 0.2;
      if (f.ringExtension > 0.5) score += 0.2;
      // Pinky curled
      if (f.pinkyCurl > 0.4) score += 0.15;
      // Spread apart
      if (f.indexMiddleSpread > 0.2) score += 0.1;
      if (f.middleRingSpread > 0.2) score += 0.1;
      return score;
    }
  },
  
  // X - Index bent like hook
  {
    letter: 'X',
    match: (f) => {
      let score = 0;
      // Index partially extended but curled (hook)
      if (f.indexExtension > 0.2 && f.indexExtension < 0.7) score += 0.25;
      if (f.indexCurl > 0.3 && f.indexCurl < 0.8) score += 0.25;
      // Other fingers curled
      if (f.middleCurl > 0.5) score += 0.15;
      if (f.ringCurl > 0.5) score += 0.1;
      if (f.pinkyCurl > 0.5) score += 0.1;
      // Thumb in
      if (f.thumbPosition !== 'out') score += 0.15;
      return score;
    }
  },
  
  // Y - Thumb and pinky extended (hang loose)
  {
    letter: 'Y',
    match: (f) => {
      let score = 0;
      // Thumb extended
      if (f.thumbExtension > 0.4) score += 0.25;
      // Pinky extended
      if (f.pinkyExtension > 0.5) score += 0.25;
      // Middle three curled
      if (f.indexCurl > 0.5) score += 0.15;
      if (f.middleCurl > 0.5) score += 0.15;
      if (f.ringCurl > 0.5) score += 0.15;
      return score;
    }
  },
  
  // Z - Index tracing Z (static = pointing with index)
  {
    letter: 'Z',
    match: (f) => {
      let score = 0;
      // Index extended pointing
      if (f.indexExtension > 0.6) score += 0.3;
      // Other fingers curled
      if (f.middleCurl > 0.4) score += 0.15;
      if (f.ringCurl > 0.4) score += 0.15;
      if (f.pinkyCurl > 0.4) score += 0.15;
      // Thumb tucked
      if (f.thumbPosition === 'tucked' || f.thumbPosition === 'across') score += 0.25;
      return score;
    }
  }
];

// ============ MAIN CLASSIFIER ============

export function classifyHandSign(landmarks: Landmark[] | undefined | null): string {
  if (!landmarks || landmarks.length < 21) {
    return '...';
  }
  
  try {
    const features = extractFeatures(landmarks);
    
    let bestMatch = { letter: '...', score: 0 };
    
    for (const template of ASL_TEMPLATES) {
      const score = template.match(features);
      if (score > bestMatch.score) {
        bestMatch = { letter: template.letter, score };
      }
    }
    
    // Require minimum confidence
    if (bestMatch.score < 0.5) {
      return '...';
    }
    
    return bestMatch.letter;
  } catch (e) {
    console.error('Classification error:', e);
    return '...';
  }
}

export function getConfidence(landmarks: Landmark[] | undefined | null, predictedLetter: string): number {
  if (!landmarks || landmarks.length < 21 || predictedLetter === '...') {
    return 0;
  }
  
  try {
    const features = extractFeatures(landmarks);
    const template = ASL_TEMPLATES.find(t => t.letter === predictedLetter);
    
    if (template) {
      return Math.min(1, template.match(features));
    }
    
    return 0.5;
  } catch (e) {
    return 0;
  }
}

// Export features for debugging
export function getHandFeatures(landmarks: Landmark[] | undefined | null): HandFeatures | null {
  if (!landmarks || landmarks.length < 21) return null;
  return extractFeatures(landmarks);
}
