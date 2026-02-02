import React, { useState, useEffect, useRef } from 'react';
import { Mic, MicOff, Play, Search, Video, SkipForward, Settings, Grid, MonitorPlay, RotateCcw } from 'lucide-react';
import { SpeechRecognition, SpeechRecognitionEvent } from '../types';
import { ASL_ALPHABET } from '../constants';

const TextToSign: React.FC = () => {
  const [inputText, setInputText] = useState('');
  const [isListening, setIsListening] = useState(false);
  const [signSequence, setSignSequence] = useState<string[]>([]);
  
  // Playback State
  const [playingWordIndex, setPlayingWordIndex] = useState<number | null>(null);
  const [playingCharIndex, setPlayingCharIndex] = useState<number | null>(null);
  const [currentImage, setCurrentImage] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(800); // ms per letter
  const [showReference, setShowReference] = useState(false);
  const [imgError, setImgError] = useState(false); // Track if image fails to load
  
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const isPlayingRef = useRef(false); // Ref to track playback status within async loops

  // Preload Images for smooth playback
  useEffect(() => {
    Object.values(ASL_ALPHABET).forEach((src) => {
        const img = new Image();
        img.src = src;
    });
  }, []);

  // Reset error state when image changes
  useEffect(() => {
    setImgError(false);
  }, [currentImage]);

  // Initialize Speech Recognition
  useEffect(() => {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (SpeechRecognition) {
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = true;
      recognitionRef.current.interimResults = true;
      recognitionRef.current.lang = 'en-US';

      recognitionRef.current.onresult = (event: SpeechRecognitionEvent) => {
        let final = '';
        for (let i = event.resultIndex; i < event.results.length; ++i) {
          if (event.results[i].isFinal) {
            final += event.results[i][0].transcript + ' ';
          }
        }
        if (final) {
          setInputText(prev => (prev + final));
        }
      };

      recognitionRef.current.onend = () => {
        if (isListening) recognitionRef.current?.start();
      };
    }
    return () => {
        if (recognitionRef.current) {
            recognitionRef.current.stop();
        }
    };
  }, [isListening]);

  const toggleMic = () => {
    if (isListening) {
      recognitionRef.current?.stop();
      setIsListening(false);
    } else {
      recognitionRef.current?.start();
      setIsListening(true);
    }
  };

  const handleTranslate = () => {
    // 1. Tokenization & Lemmatization
    // STRICT FILTER: Only allow a-z characters (and whitespace).
    const words = inputText
      .toLowerCase()
      .replace(/[^a-z\s]/g, "") 
      .split(/\s+/)
      .filter(w => w.length > 0);
    
    setSignSequence(words);
    stopPlayback();
    setShowReference(false);
  };

  const stopPlayback = () => {
    setIsPlaying(false);
    isPlayingRef.current = false;
    setPlayingWordIndex(null);
    setPlayingCharIndex(null);
    setCurrentImage(null);
  };

  const playSequence = async () => {
    if (signSequence.length === 0 || isPlayingRef.current) return;
    
    setIsPlaying(true);
    isPlayingRef.current = true;
    setShowReference(false);
    
    // Loop through Words
    for (let w = 0; w < signSequence.length; w++) {
      if (!isPlayingRef.current) break;
      
      const word = signSequence[w];
      setPlayingWordIndex(w);
      setPlayingCharIndex(null); // Reset char index for new word
      
      // Loop through Characters (Fingerspelling)
      for (let c = 0; c < word.length; c++) {
        if (!isPlayingRef.current) break;

        const char = word[c];
        const imageUrl = ASL_ALPHABET[char];
        
        setPlayingCharIndex(c);
        
        if (imageUrl) {
            setCurrentImage(imageUrl);
        } else {
            setCurrentImage(null);
        }

        // Use dynamic speed
        await new Promise(resolve => setTimeout(resolve, playbackSpeed));
      }
      
      // Short pause between words
      setCurrentImage(null);
      if (isPlayingRef.current) {
          await new Promise(resolve => setTimeout(resolve, playbackSpeed * 0.5));
      }
    }
    
    stopPlayback();
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-12 gap-4 sm:gap-6 h-full">
      {/* Left Panel: Input */}
      <div className="md:col-span-5 flex flex-col gap-3 sm:gap-4">
        <div className="bg-slate-800 rounded-lg sm:rounded-xl p-4 sm:p-6 border border-slate-700 shadow-lg flex flex-col gap-3 sm:gap-4 min-h-[250px] md:h-full">
          <div className="flex items-center justify-between">
            <h3 className="text-white font-semibold flex items-center gap-2 text-sm sm:text-base">
              <Search className="w-4 h-4 sm:w-5 sm:h-5 text-emerald-400" />
              Input Text or Speech
            </h3>
            <span className="text-[10px] sm:text-xs text-slate-500 uppercase tracking-wider font-bold">Step 1</span>
          </div>

          <textarea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Type here or use microphone (Only A-Z supported)..."
            className="flex-1 min-h-[120px] bg-slate-900/50 border border-slate-700 rounded-lg p-3 sm:p-4 text-slate-200 focus:outline-none focus:ring-2 focus:ring-emerald-500/50 resize-none font-medium text-base sm:text-lg"
          />

          <div className="flex gap-2 sm:gap-3">
            <button
              onClick={toggleMic}
              className={`flex-1 py-2.5 sm:py-3 px-2 sm:px-4 rounded-lg font-semibold flex items-center justify-center gap-1.5 sm:gap-2 transition-all text-sm sm:text-base ${
                isListening
                  ? 'bg-red-500/20 text-red-400 border border-red-500/50 animate-pulse'
                  : 'bg-slate-700 hover:bg-slate-600 text-white'
              }`}
            >
              {isListening ? <MicOff className="w-4 h-4 sm:w-5 sm:h-5" /> : <Mic className="w-4 h-4 sm:w-5 sm:h-5" />}
              <span className="hidden sm:inline">{isListening ? 'Stop Mic' : 'Voice Input'}</span>
              <span className="sm:hidden">{isListening ? 'Stop' : 'Voice'}</span>
            </button>
            <button
              onClick={handleTranslate}
              className="flex-1 py-2.5 sm:py-3 px-2 sm:px-4 rounded-lg font-semibold bg-emerald-600 hover:bg-emerald-500 text-white shadow-lg shadow-emerald-500/20 transition-all text-sm sm:text-base"
            >
              Process
            </button>
          </div>
        </div>
      </div>

      {/* Right Panel: Video Output */}
      <div className="md:col-span-7 flex flex-col gap-3 sm:gap-4">
        <div className="bg-slate-900 rounded-lg sm:rounded-xl border border-slate-800 overflow-hidden shadow-2xl relative flex flex-col h-[400px] sm:h-[500px] md:h-[600px]">
          <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-emerald-500 to-blue-500"></div>
          
          {/* Main Viewer */}
          <div className="flex-1 bg-black relative flex flex-col items-center justify-center p-6 overflow-hidden">
            
            {showReference ? (
                 <div className="w-full h-full overflow-y-auto custom-scrollbar">
                    <h3 className="text-white font-bold mb-4 sticky top-0 bg-black/90 p-2 border-b border-slate-800 flex items-center gap-2">
                        <Grid className="w-4 h-4 text-emerald-400"/> ASL Alphabet Reference
                    </h3>
                    <div className="grid grid-cols-4 sm:grid-cols-5 md:grid-cols-6 gap-2">
                        {Object.entries(ASL_ALPHABET).map(([letter, src]) => (
                            <div key={letter} className="bg-slate-800 p-2 rounded border border-slate-700 flex flex-col items-center hover:bg-slate-700 transition-colors">
                                <div className="bg-white rounded p-1 w-full aspect-square flex items-center justify-center">
                                    <img src={src} alt={letter} className="w-full h-full object-contain" loading="lazy" />
                                </div>
                                <span className="text-xs font-bold text-slate-400 mt-1 uppercase">{letter}</span>
                            </div>
                        ))}
                    </div>
                 </div>
            ) : signSequence.length > 0 ? (
              isPlaying ? (
                <div className="flex flex-col items-center justify-center w-full h-full">
                    
                    {/* Full Sentence Display (Moved to Top) */}
                    <div className="w-full px-4 md:px-8 bg-slate-900/50 p-6 rounded-xl border border-slate-800/50 backdrop-blur-sm mb-8">
                        <div className="flex flex-wrap justify-center gap-x-6 gap-y-4 text-2xl font-bold leading-relaxed">
                             {signSequence.map((word, wIdx) => (
                                 <div key={wIdx} className={`flex transition-all duration-300 ${playingWordIndex === wIdx ? 'scale-105' : 'text-slate-600'}`}>
                                    {word.split('').map((char, cIdx) => (
                                         <span 
                                            key={cIdx} 
                                            className={`
                                                uppercase inline-block transition-all duration-100 px-0.5 opacity-100
                                                ${playingWordIndex === wIdx && playingCharIndex === cIdx 
                                                    ? 'text-emerald-400 -translate-y-2 scale-125 drop-shadow-[0_0_10px_rgba(52,211,153,0.6)]' 
                                                    : (playingWordIndex === wIdx ? 'text-white' : 'text-slate-500')}
                                            `}
                                         >
                                             {char}
                                         </span>
                                    ))}
                                 </div>
                             ))}
                        </div>
                        <p className="text-emerald-500/50 font-mono text-[10px] uppercase tracking-[0.3em] text-center mt-4">
                            Active Translation Sequence
                        </p>
                    </div>

                    {/* The Image Container - White Background for visibility */}
                    <div className="relative w-72 h-72 bg-white rounded-3xl border-4 border-slate-200 flex items-center justify-center overflow-hidden shadow-2xl ring-4 ring-slate-800/50">
                        {currentImage && !imgError ? (
                            <img 
                                src={currentImage} 
                                alt="Sign Language Gesture" 
                                className="w-full h-full object-contain"
                                style={{ opacity: 1 }} // Inline style to enforce 100% opacity
                                onError={() => setImgError(true)}
                            />
                        ) : (
                            // Fallback / Loading State
                            <div className="flex flex-col items-center justify-center w-full h-full bg-slate-50">
                                {isPlaying && playingWordIndex !== null && playingCharIndex !== null ? (
                                    imgError ? (
                                        // Fallback if image file is missing
                                        <div className="flex flex-col items-center animate-in fade-in">
                                            <span className="text-9xl font-black text-slate-300 select-none">
                                                {signSequence[playingWordIndex][playingCharIndex].toUpperCase()}
                                            </span>
                                            <div className="mt-2 bg-red-100 text-red-500 text-[10px] font-mono px-2 py-1 rounded border border-red-200">
                                                IMAGE MISSING
                                            </div>
                                        </div>
                                    ) : (
                                        // Loading
                                        <div className="w-12 h-12 border-4 border-slate-200 border-t-emerald-500 rounded-full animate-spin"></div>
                                    )
                                ) : (
                                     // Idle
                                     <div className="w-12 h-12 border-4 border-slate-200 border-t-emerald-500 rounded-full animate-spin"></div>
                                )}
                            </div>
                        )}
                        
                         {/* Current Letter Overlay - Always visible */}
                        {playingWordIndex !== null && playingCharIndex !== null && (
                            <div className="absolute bottom-3 right-3 bg-emerald-600 text-white w-12 h-12 flex items-center justify-center rounded-2xl font-black text-2xl shadow-lg border-2 border-white z-10 opacity-100">
                                {signSequence[playingWordIndex][playingCharIndex].toUpperCase()}
                            </div>
                        )}
                    </div>

                </div>
              ) : (
                <div className="text-center text-slate-500">
                  <Video className="w-16 h-16 mx-auto mb-4 opacity-50" />
                  <p className="text-lg mb-2 text-slate-300 font-medium">Ready to Animate</p>
                  <p className="text-xs text-slate-500 max-w-xs mx-auto mb-6">Sequence loaded with {signSequence.length} words. Press play to view the sign language translation.</p>
                  
                  <div className="flex justify-center gap-4">
                    <button 
                        onClick={() => { setSignSequence([]); setInputText(''); }}
                        className="px-6 py-4 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-full font-bold transition-all flex items-center gap-2"
                    >
                        <RotateCcw className="w-5 h-5" />
                        Reset
                    </button>
                    <button 
                        onClick={playSequence}
                        className="px-10 py-4 bg-blue-600 hover:bg-blue-500 text-white rounded-full font-bold shadow-lg shadow-blue-500/25 transition-all flex items-center gap-2 group hover:scale-105"
                    >
                        <Play className="w-5 h-5 fill-current" />
                        Play Animation
                    </button>
                  </div>
                </div>
              )
            ) : (
              <div className="text-center text-slate-600">
                <MonitorPlay className="w-24 h-24 mx-auto mb-4 opacity-20" />
                <p>Waiting for text to generate signs...</p>
              </div>
            )}
            
            {/* Playback Controls Overlay */}
            <div className="absolute top-4 right-4 flex gap-2">
                <button 
                    onClick={() => setShowReference(!showReference)}
                    className={`p-2 rounded-lg border backdrop-blur-md transition-all ${showReference ? 'bg-emerald-500/20 border-emerald-500/50 text-emerald-300' : 'bg-slate-900/80 border-slate-700 text-slate-400 hover:text-white'}`}
                    title="Toggle Reference Grid"
                >
                    <Grid className="w-4 h-4" />
                </button>
            </div>

            {/* Speed Control Overlay */}
            {signSequence.length > 0 && !isPlaying && !showReference && (
                <div className="absolute bottom-4 left-0 right-0 flex justify-center gap-2">
                     <div className="bg-slate-900/90 border border-slate-700 rounded-full p-1 flex items-center">
                        <Settings className="w-4 h-4 text-slate-500 ml-3 mr-2" />
                        <button onClick={() => setPlaybackSpeed(1200)} className={`px-3 py-1 text-xs font-bold rounded-full transition-colors ${playbackSpeed === 1200 ? 'bg-slate-700 text-white' : 'text-slate-400 hover:text-white'}`}>Slow</button>
                        <button onClick={() => setPlaybackSpeed(800)} className={`px-3 py-1 text-xs font-bold rounded-full transition-colors ${playbackSpeed === 800 ? 'bg-slate-700 text-white' : 'text-slate-400 hover:text-white'}`}>Normal</button>
                        <button onClick={() => setPlaybackSpeed(400)} className={`px-3 py-1 text-xs font-bold rounded-full transition-colors ${playbackSpeed === 400 ? 'bg-slate-700 text-white' : 'text-slate-400 hover:text-white'}`}>Fast</button>
                     </div>
                </div>
            )}
          </div>

          {/* Sequence Timeline (Small) */}
          <div className="h-16 bg-slate-950 border-t border-slate-800 flex items-center px-4 gap-2 overflow-x-auto custom-scrollbar">
            {signSequence.map((word, idx) => (
               <span key={idx} className={`text-xs px-2 py-1 rounded ${playingWordIndex === idx ? 'bg-emerald-900 text-emerald-400' : 'text-slate-600'}`}>
                   {word}
               </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default TextToSign;