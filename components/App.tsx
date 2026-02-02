import React, { useState } from 'react';
import { Hand, Type, FileVideo, Database } from 'lucide-react';
import { SpeedInsights } from '@vercel/speed-insights/react';
import SignToText from './SignToText';
import TextToSign from './TextToSign';
import DatasetCapture from './DatasetCapture';


type Tab = 'sign-to-text' | 'text-to-sign' | 'capture-dataset';

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<Tab>('sign-to-text');

  return (
    <div className="min-h-screen flex flex-col bg-slate-950 text-slate-100 font-sans selection:bg-blue-500/30">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-950/80 backdrop-blur-md sticky top-0 z-50 safe-area-top">
        <div className="max-w-7xl mx-auto px-3 sm:px-4 h-14 sm:h-16 flex items-center justify-between">
          <div className="flex items-center gap-2 sm:gap-3 flex-shrink-0">
            <div className="bg-gradient-to-br from-blue-600 to-cyan-500 p-1.5 sm:p-2 rounded-lg shadow-lg shadow-blue-500/20">
              <Hand className="w-5 h-5 sm:w-6 sm:h-6 text-white" />
            </div>
            <div className="hidden xs:block">
                <h1 className="text-base sm:text-xl font-bold tracking-tight bg-gradient-to-r from-blue-100 to-slate-300 bg-clip-text text-transparent">
                SignBridge AI
                </h1>
                <p className="text-[8px] sm:text-[10px] text-slate-500 font-mono tracking-widest uppercase hidden sm:block">Bi-Directional Translation</p>
            </div>
          </div>
          
          <nav className="flex gap-0.5 sm:gap-1 bg-slate-900/50 p-0.5 sm:p-1 rounded-lg border border-slate-800 overflow-x-auto hide-scrollbar">
            <button
              onClick={() => setActiveTab('sign-to-text')}
              className={`px-2 sm:px-5 py-1.5 sm:py-2 rounded-md text-xs sm:text-sm font-bold transition-all duration-200 flex items-center gap-1 sm:gap-2 whitespace-nowrap ${
                activeTab === 'sign-to-text'
                  ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/25'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800'
              }`}
            >
              <Hand className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
              <span className="hidden sm:inline">Sign to Text</span>
              <span className="sm:hidden">Sign</span>
            </button>
            <button
              onClick={() => setActiveTab('text-to-sign')}
              className={`px-2 sm:px-5 py-1.5 sm:py-2 rounded-md text-xs sm:text-sm font-bold transition-all duration-200 flex items-center gap-1 sm:gap-2 whitespace-nowrap ${
                activeTab === 'text-to-sign'
                  ? 'bg-emerald-600 text-white shadow-lg shadow-emerald-500/25'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800'
              }`}
            >
              <Type className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
              <span className="hidden sm:inline">Text to Sign</span>
              <span className="sm:hidden">Text</span>
            </button>
            <button
              onClick={() => setActiveTab('capture-dataset')}
              className={`px-2 sm:px-5 py-1.5 sm:py-2 rounded-md text-xs sm:text-sm font-bold transition-all duration-200 flex items-center gap-1 sm:gap-2 whitespace-nowrap ${
                activeTab === 'capture-dataset'
                  ? 'bg-purple-600 text-white shadow-lg shadow-purple-500/25'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800'
              }`}
            >
              <Database className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
              <span className="hidden sm:inline">Capture</span>
              <span className="sm:hidden">Cap</span>
            </button>
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-2 sm:p-4 md:p-6 lg:p-8">
        <div className="bg-slate-900/40 border border-slate-800 rounded-xl sm:rounded-2xl p-0.5 sm:p-1 min-h-[calc(100vh-180px)] sm:min-h-[600px] shadow-2xl backdrop-blur-sm overflow-hidden relative ring-1 ring-white/5">
          
          {/* Background decoration */}
          <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-blue-500/5 rounded-full blur-3xl -z-10 pointer-events-none"></div>
          <div className="absolute bottom-0 left-0 w-[500px] h-[500px] bg-emerald-500/5 rounded-full blur-3xl -z-10 pointer-events-none"></div>

          <div className="p-6 h-full">
            {activeTab === 'sign-to-text' && (
              <div className="animate-in fade-in zoom-in-95 duration-500">
                <SignToText />
              </div>
            )}

            {activeTab === 'text-to-sign' && (
              <div className="animate-in fade-in zoom-in-95 duration-500">
                <TextToSign />
              </div>
            )}

            {activeTab === 'capture-dataset' && (
              <div className="animate-in fade-in zoom-in-95 duration-500">
                <DatasetCapture />
              </div>
            )}
          </div>

        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/50 py-2 sm:py-4 text-center text-slate-500 text-[10px] sm:text-xs bg-slate-950/80 backdrop-blur-sm font-mono safe-area-bottom">
        <div className="flex items-center justify-center gap-2 sm:gap-4 flex-wrap px-2">
            <span className="flex items-center gap-1 sm:gap-2">
                <span className="w-1 h-1 sm:w-1.5 sm:h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
                <span className="hidden sm:inline">SYSTEM_READY</span>
                <span className="sm:hidden">READY</span>
            </span>
            <span className="text-slate-700 hidden sm:inline">|</span>
            <span>V 2.4.0</span>
            <span className="text-slate-700 hidden sm:inline">|</span>
            <span className="hidden sm:inline">SECURE CONNECTION</span>
        </div>
      </footer>
      <SpeedInsights />
    </div>
  );
};

export default App;
