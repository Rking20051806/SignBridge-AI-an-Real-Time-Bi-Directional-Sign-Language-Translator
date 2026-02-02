import { GoogleGenAI, Type, HarmCategory, HarmBlockThreshold } from "@google/genai";
import { DetectionResult } from "../types";

const getClient = () => {
  return new GoogleGenAI({ apiKey: import.meta.env.VITE_GEMINI_API_KEY });
};

/**
 * Optimized for Single Letter Recognition (A-Z)
 * Expects a cropped image of a hand for maximum accuracy.
 */
export const analyzeSignLanguageFrame = async (base64Image: string): Promise<string | null> => {
  try {
    const ai = getClient();
    const cleanBase64 = base64Image.replace(/^data:image\/\w+;base64,/, '');

    const response = await ai.models.generateContent({
      model: 'gemini-3-flash-preview',
      contents: {
        parts: [
            {
                inlineData: {
                    mimeType: 'image/jpeg',
                    data: cleanBase64
                }
            },
            {
                text: `You are a strict ASL Alphabet Classifier.
                
                Task: Identify the American Sign Language (ASL) letter shown in this cropped hand image.
                
                Constraints:
                1. Output ONLY a single letter (A-Z).
                2. Do not output words (e.g., if it looks like "Hello", find the closest letter or return "...").
                3. If the image is blurry, ambiguous, or not a hand, return "...".
                
                Response format: Just the letter.`
            }
        ]
      },
      config: {
        safetySettings: [
            { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_NONE },
            { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_NONE },
            { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_NONE },
            { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_NONE },
        ]
      }
    });

    const text = response.text?.trim().toUpperCase();
    
    // Strict client-side validation
    if (text && text.length === 1 && /[A-Z]/.test(text)) {
        return text;
    }
    
    return "...";

  } catch (error: any) {
    return null;
  }
};

/**
 * Deep Analysis: Scans the entire image for ALL visible hand signs.
 */
export const detectSignWithBoundingBox = async (base64Image: string): Promise<DetectionResult> => {
    try {
      const ai = getClient();
      const cleanBase64 = base64Image.replace(/^data:image\/\w+;base64,/, '');
  
      const response = await ai.models.generateContent({
        model: 'gemini-3-flash-preview',
        contents: {
          parts: [
              {
                  inlineData: {
                      mimeType: 'image/jpeg',
                      data: cleanBase64
                  }
              },
              {
                  text: `Analyze this image for ASL Alphabet Hand Signs (A-Z).
                  
                  Task:
                  1. Detect every individual hand sign.
                  2. Identify the specific letter (A-Z).
                  3. Return the bounding box [ymin, xmin, ymax, xmax] (0-1000 scale).
                  
                  Output Requirements:
                  - List ALL detections.
                  - Label must be a single letter A-Z.`
              }
          ]
        },
        config: {
            responseMimeType: "application/json",
            responseSchema: {
                type: Type.OBJECT,
                properties: {
                    summary: { type: Type.STRING },
                    detections: {
                        type: Type.ARRAY,
                        items: {
                            type: Type.OBJECT,
                            properties: {
                                label: { type: Type.STRING },
                                box_2d: { 
                                    type: Type.ARRAY, 
                                    items: { type: Type.INTEGER } 
                                }
                            },
                            required: ["label", "box_2d"]
                        }
                    }
                },
                required: ["summary", "detections"],
            }
        }
      });
  
      const text = response.text;
      if (!text) throw new Error("No result");
  
      const parsed = JSON.parse(text);
      return {
          summary: parsed.summary || `Detected ${parsed.detections?.length || 0} signs`,
          detections: Array.isArray(parsed.detections) ? parsed.detections : []
      };

    } catch (error: any) {
      console.error("Multi-Detection Failure:", error);
      throw error;
    }
  };