import React, { useState, useEffect, useRef } from 'react';
import { Mic, MicOff, Type, Volume2, Trash2 } from 'lucide-react';
import { SpeechRecognition, SpeechRecognitionEvent, SpeechRecognitionErrorEvent } from '../types';

const HearingToDeaf: React.FC = () => {
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState<string>('');
  const [interimTranscript, setInterimTranscript] = useState<string>('');
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const [fontSize, setFontSize] = useState<number>(32);

  useEffect(() => {
    // Initialize Speech Recognition
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    
    if (SpeechRecognition) {
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = true;
      recognitionRef.current.interimResults = true;
      recognitionRef.current.lang = 'en-US';

      recognitionRef.current.onresult = (event: SpeechRecognitionEvent) => {
        let final = '';
        let interim = '';

        for (let i = event.resultIndex; i < event.results.length; ++i) {
          if (event.results[i].isFinal) {
            final += event.results[i][0].transcript + ' ';
          } else {
            interim += event.results[i][0].transcript;
          }
        }

        if (final) {
          setTranscript(prev => (prev + final).slice(-1000)); // Keep last 1000 chars to avoid memory issues
        }
        setInterimTranscript(interim);
      };

      recognitionRef.current.onerror = (event: SpeechRecognitionErrorEvent) => {
        console.error("Speech recognition error", event.error);
        if (event.error === 'not-allowed') {
            setIsListening(false);
        }
      };
      
      recognitionRef.current.onend = () => {
          // Auto restart if still supposed to be listening
          if (isListening) {
             try {
                recognitionRef.current?.start();
             } catch {
                setIsListening(false);
             }
          }
      };
    } else {
      alert("Your browser does not support Speech Recognition. Please use Chrome.");
    }

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
    };
  }, [isListening]);

  const toggleListening = () => {
    if (isListening) {
      recognitionRef.current?.stop();
      setIsListening(false);
    } else {
      try {
          recognitionRef.current?.start();
          setIsListening(true);
      } catch (e) {
          console.error(e);
      }
    }
  };

  const clearTranscript = () => {
    setTranscript('');
    setInterimTranscript('');
  };

  return (
    <div className="flex flex-col h-full gap-6">
        {/* Controls Header */}
        <div className="flex flex-wrap items-center justify-between gap-4 bg-slate-800 p-4 rounded-xl border border-slate-700">
            <button
                onClick={toggleListening}
                className={`flex items-center gap-3 px-8 py-4 rounded-xl font-bold text-lg transition-all shadow-lg ${
                    isListening 
                    ? 'bg-red-500 hover:bg-red-600 text-white shadow-red-500/20 animate-pulse' 
                    : 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-emerald-500/20'
                }`}
            >
                {isListening ? <MicOff className="w-6 h-6" /> : <Mic className="w-6 h-6" />}
                {isListening ? 'Stop Listening' : 'Start Listening'}
            </button>

            <div className="flex items-center gap-4">
                <div className="flex items-center gap-2 bg-slate-700/50 p-2 rounded-lg">
                    <Type className="w-4 h-4 text-slate-400" />
                    <input 
                        type="range" 
                        min="16" 
                        max="72" 
                        value={fontSize} 
                        onChange={(e) => setFontSize(Number(e.target.value))}
                        className="w-32 accent-blue-500"
                    />
                    <span className="text-xs text-slate-400 w-6">{fontSize}px</span>
                </div>
                <button 
                    onClick={clearTranscript}
                    className="p-3 text-slate-400 hover:text-red-400 hover:bg-slate-700 rounded-lg transition-colors"
                    title="Clear Text"
                >
                    <Trash2 className="w-5 h-5" />
                </button>
            </div>
        </div>

        {/* Visualizer / Status */}
        {isListening && (
            <div className="flex items-center gap-2 justify-center text-emerald-400 text-sm font-mono">
                <Volume2 className="w-4 h-4 animate-bounce" />
                <span>Microphone active - Speak clearly</span>
            </div>
        )}

        {/* Display Area */}
        <div className="flex-1 bg-black rounded-2xl p-8 border-2 border-slate-700 shadow-inner overflow-y-auto min-h-[400px] flex flex-col">
            {transcript === '' && interimTranscript === '' ? (
                <div className="flex-1 flex flex-col items-center justify-center text-slate-600 opacity-50">
                    <Mic className="w-24 h-24 mb-4" />
                    <p className="text-xl">Waiting for speech...</p>
                </div>
            ) : (
                <div className="text-left font-sans leading-relaxed transition-all duration-200" style={{ fontSize: `${fontSize}px` }}>
                    <span className="text-white">{transcript}</span>
                    <span className="text-yellow-400">{interimTranscript}</span>
                </div>
            )}
        </div>
    </div>
  );
};

export default HearingToDeaf;