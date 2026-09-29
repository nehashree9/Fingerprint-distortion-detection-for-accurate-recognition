import React, { useState, useRef } from 'react';
import { Fingerprint, UploadCloud, CheckCircle2, AlertCircle, RefreshCw, Layers, ShieldCheck, FileCheck, Search, Activity, Image as ImageIcon } from 'lucide-react';
import clsx from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: (string | undefined | null | false)[]) {
  return twMerge(clsx(inputs));
}

type AuthResult = {
  authenticated: boolean;
  identity_code: string;
  distortion_probability: number;
  is_distorted: boolean;
  rectification_applied: boolean;
  best_score: number;
  best_raw_score: number;
  best_impression_id: number;
  original_image_url: string | null;
  rectified_image_url: string | null;
  scores: Array<{
    impression_id: number;
    score: number;
    raw_score: number;
    decision: boolean;
  }>;
};

export default function App() {
  const [identityCode, setIdentityCode] = useState('');
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<AuthResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedImage(file);
      const url = URL.createObjectURL(file);
      setImagePreview(url);
      setResult(null);
      setError(null);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedImage(file);
      const url = URL.createObjectURL(file);
      setImagePreview(url);
      setResult(null);
      setError(null);
    }
  };

  const handleAuthenticate = async () => {
    if (!selectedImage) {
      setError('Please select a fingerprint image.');
      return;
    }
    if (!identityCode.trim()) {
      setError('Please enter an identity code.');
      return;
    }

    setIsProcessing(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('identity_code', identityCode);
    formData.append('fingerprint', selectedImage);

    try {
      const response = await fetch('http://127.0.0.1:8000/authenticate', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Authentication failed on the server.');
      }

      const data: AuthResult = await response.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || 'An unexpected error occurred.');
    } finally {
      setIsProcessing(false);
    }
  };

  const getProcessingStages = () => {
    if (!result) return [];
    
    const stages = [
      { name: 'Image Uploaded', status: 'completed' },
      { name: 'Distortion Detection', status: 'completed', detail: `Prob: ${(result.distortion_probability * 100).toFixed(1)}%` },
    ];
    
    if (result.is_distorted) {
      stages.push({ name: 'DDRNet Rectification', status: 'completed', detail: 'Applied' });
    } else {
      stages.push({ name: 'Clean Fingerprint', status: 'skipped', detail: 'Rectification skipped' });
    }
    
    stages.push(
      { name: 'MINDTCT Minutiae', status: 'completed' },
      { name: 'Bozorth3 Matching', status: 'completed', detail: `Best score: ${result.best_score.toFixed(4)}` },
      { name: 'Authentication', status: result.authenticated ? 'success' : 'failed' }
    );
    
    return stages;
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 font-sans selection:bg-blue-900 selection:text-blue-100">
      {/* Header */}
      <header className="bg-slate-900 border-b border-slate-800 sticky top-0 z-10 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-blue-600 p-2 rounded-lg">
              <Fingerprint className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-slate-100 tracking-tight leading-tight">Fingerprint Authentication System</h1>
              <p className="text-xs text-slate-400 font-medium tracking-wide">Distortion Detection • DDRNet Rectification • Minutiae Matching</p>
            </div>
          </div>
          <div className="hidden sm:flex items-center text-sm font-medium text-slate-400 gap-2 bg-slate-800/50 px-3 py-1.5 rounded-full border border-slate-700/50">
            <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div>
            System Online
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        
        {/* Workspace Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* Left Column: Input */}
          <div className="lg:col-span-5 space-y-6">
            
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl relative overflow-hidden">
              <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-blue-500 to-indigo-500"></div>
              <h2 className="text-lg font-semibold text-slate-100 mb-6 flex items-center gap-2">
                <FileCheck className="w-5 h-5 text-blue-400" />
                Input Fingerprint
              </h2>
              
              <div className="space-y-6">
                
                {/* Upload Area */}
                <div>
                  <label className="block text-sm font-medium text-slate-400 mb-2">Fingerprint Image</label>
                  {!imagePreview ? (
                    <div 
                      onDragOver={handleDragOver}
                      onDrop={handleDrop}
                      className="border-2 border-dashed border-slate-700 hover:border-blue-500/50 hover:bg-slate-800/50 transition-colors rounded-xl p-8 flex flex-col items-center justify-center text-center cursor-pointer group h-64"
                      onClick={() => fileInputRef.current?.click()}
                    >
                      <div className="bg-slate-800 p-4 rounded-full mb-4 group-hover:bg-blue-900/30 group-hover:text-blue-400 transition-colors">
                        <UploadCloud className="w-8 h-8 text-slate-400 group-hover:text-blue-400" />
                      </div>
                      <p className="text-slate-300 font-medium mb-1">Click to upload or drag and drop</p>
                      <p className="text-xs text-slate-500">PNG, JPG or JPEG</p>
                    </div>
                  ) : (
                    <div className="relative rounded-xl border border-slate-700 overflow-hidden bg-slate-950 flex flex-col items-center justify-center h-64 group">
                      <img src={imagePreview} alt="Fingerprint preview" className="max-h-full max-w-full object-contain p-2" />
                      <div className="absolute inset-0 bg-slate-950/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center backdrop-blur-sm">
                        <button 
                          onClick={() => fileInputRef.current?.click()}
                          className="bg-slate-800 hover:bg-slate-700 text-white px-4 py-2 rounded-lg font-medium text-sm transition-colors border border-slate-600 shadow-lg"
                        >
                          Change Image
                        </button>
                      </div>
                      <div className="absolute bottom-2 left-2 right-2 bg-slate-900/90 backdrop-blur-md px-3 py-2 rounded-lg border border-slate-700/50 flex items-center justify-between pointer-events-none">
                        <span className="text-xs font-medium text-slate-300 truncate max-w-[150px]">{selectedImage?.name}</span>
                        <span className="text-xs text-slate-500">{(selectedImage?.size ? (selectedImage.size / 1024).toFixed(1) : 0)} KB</span>
                      </div>
                    </div>
                  )}
                  <input 
                    type="file" 
                    ref={fileInputRef} 
                    className="hidden" 
                    accept="image/png, image/jpeg, image/jpg"
                    onChange={handleImageChange}
                  />
                </div>

                {/* Identity Input */}
                <div>
                  <label htmlFor="identity" className="block text-sm font-medium text-slate-400 mb-2">Identity Code</label>
                  <div className="relative">
                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                      <Search className="h-4 w-4 text-slate-500" />
                    </div>
                    <input
                      type="text"
                      id="identity"
                      value={identityCode}
                      onChange={(e) => setIdentityCode(e.target.value)}
                      placeholder="e.g. FAMILY-13/CHILD"
                      className="block w-full pl-10 bg-slate-950 border border-slate-700 rounded-xl py-2.5 text-slate-200 placeholder-slate-500 focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500 transition-colors sm:text-sm"
                    />
                  </div>
                </div>
                
                {error && (
                  <div className="bg-red-950/50 border border-red-900/50 text-red-400 px-4 py-3 rounded-xl flex items-start gap-3 text-sm">
                    <AlertCircle className="w-5 h-5 shrink-0 mt-0.5" />
                    <p>{error}</p>
                  </div>
                )}

                {/* Authenticate Button */}
                <button
                  onClick={handleAuthenticate}
                  disabled={isProcessing || !selectedImage || !identityCode}
                  className="w-full relative group overflow-hidden bg-blue-600 hover:bg-blue-500 text-white font-semibold py-3.5 px-4 rounded-xl transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-blue-900/20"
                >
                  <span className={cn("relative z-10 flex items-center justify-center gap-2", isProcessing ? "opacity-0" : "opacity-100")}>
                    <ShieldCheck className="w-5 h-5" />
                    Authenticate Fingerprint
                  </span>
                  
                  {isProcessing && (
                    <span className="absolute inset-0 z-10 flex items-center justify-center gap-3">
                      <RefreshCw className="w-5 h-5 animate-spin" />
                      Processing...
                    </span>
                  )}
                </button>
                
              </div>
            </div>
            
            {/* Pipeline Visualization (Shows during/after processing) */}
            {(isProcessing || result) && (
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl relative overflow-hidden">
                <h3 className="text-sm font-semibold text-slate-100 mb-4 flex items-center gap-2 uppercase tracking-wider">
                  <Activity className="w-4 h-4 text-indigo-400" />
                  Processing Pipeline
                </h3>
                <div className="space-y-3">
                  {isProcessing ? (
                    <div className="flex items-center gap-3 text-blue-400 text-sm font-medium">
                      <RefreshCw className="w-4 h-4 animate-spin" />
                      Running authentication pipeline...
                    </div>
                  ) : (
                    getProcessingStages().map((stage, idx) => (
                      <div key={idx} className="flex items-start gap-3">
                        <div className="mt-0.5">
                          {stage.status === 'completed' && <CheckCircle2 className="w-4 h-4 text-emerald-500" />}
                          {stage.status === 'skipped' && <RefreshCw className="w-4 h-4 text-slate-500 rotate-90" />}
                          {stage.status === 'success' && <ShieldCheck className="w-4 h-4 text-emerald-500" />}
                          {stage.status === 'failed' && <AlertCircle className="w-4 h-4 text-red-500" />}
                        </div>
                        <div>
                          <p className={cn(
                            "text-sm font-medium leading-tight",
                            stage.status === 'success' ? "text-emerald-400" :
                            stage.status === 'failed' ? "text-red-400" :
                            stage.status === 'skipped' ? "text-slate-400" : "text-slate-200"
                          )}>
                            {stage.name}
                          </p>
                          {stage.detail && <p className="text-xs text-slate-500 mt-0.5">{stage.detail}</p>}
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>
            )}
            
          </div>

          {/* Right Column: Results */}
          <div className="lg:col-span-7 space-y-6">
            
            {result ? (
              <>
                {/* Result Status Card */}
                <div className={cn(
                  "rounded-2xl p-6 shadow-xl relative overflow-hidden border",
                  result.authenticated 
                    ? "bg-slate-900 border-emerald-500/30" 
                    : "bg-slate-900 border-red-500/30"
                )}>
                  <div className={cn(
                    "absolute top-0 left-0 w-full h-1",
                    result.authenticated ? "bg-emerald-500" : "bg-red-500"
                  )}></div>
                  
                  <div className="flex items-start justify-between mb-8">
                    <div>
                      <p className="text-sm font-medium text-slate-400 mb-1 uppercase tracking-wider">Authentication Status</p>
                      <h2 className={cn(
                        "text-3xl font-bold tracking-tight",
                        result.authenticated ? "text-emerald-400" : "text-red-400"
                      )}>
                        {result.authenticated ? "AUTHENTICATED" : "NOT AUTHENTICATED"}
                      </h2>
                    </div>
                    <div className={cn(
                      "p-3 rounded-xl border",
                      result.authenticated ? "bg-emerald-500/10 border-emerald-500/20 text-emerald-400" : "bg-red-500/10 border-red-500/20 text-red-400"
                    )}>
                      {result.authenticated ? <ShieldCheck className="w-8 h-8" /> : <AlertCircle className="w-8 h-8" />}
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                    <div className="bg-slate-950/50 p-3 rounded-xl border border-slate-800">
                      <p className="text-xs text-slate-500 font-medium mb-1">Identity</p>
                      <p className="text-sm font-semibold text-slate-200 truncate">{result.identity_code}</p>
                    </div>
                    <div className="bg-slate-950/50 p-3 rounded-xl border border-slate-800">
                      <p className="text-xs text-slate-500 font-medium mb-1">Distortion Prob</p>
                      <p className="text-sm font-semibold text-slate-200">{(result.distortion_probability * 100).toFixed(1)}%</p>
                    </div>
                    <div className="bg-slate-950/50 p-3 rounded-xl border border-slate-800">
                      <p className="text-xs text-slate-500 font-medium mb-1">Distorted</p>
                      <p className={cn("text-sm font-semibold", result.is_distorted ? "text-amber-400" : "text-emerald-400")}>
                        {result.is_distorted ? "Yes" : "No"}
                      </p>
                    </div>
                    <div className="bg-slate-950/50 p-3 rounded-xl border border-slate-800">
                      <p className="text-xs text-slate-500 font-medium mb-1">Rectification</p>
                      <p className="text-sm font-semibold text-slate-200">{result.rectification_applied ? "Applied" : "Skipped"}</p>
                    </div>
                  </div>
                </div>

                {/* Images Row */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                  {/* Input Image */}
                  <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl">
                    <h3 className="text-sm font-semibold text-slate-300 mb-4 flex items-center gap-2">
                      <ImageIcon className="w-4 h-4 text-blue-400" />
                      Input Fingerprint
                    </h3>
                    <div className="bg-slate-950 rounded-xl border border-slate-800 flex items-center justify-center p-4 h-64 overflow-hidden relative">
                      {result.original_image_url ? (
                        <img 
                          src={`http://127.0.0.1:8000${result.original_image_url}`} 
                          alt="Original Input" 
                          className="max-w-full max-h-full object-contain filter drop-shadow-md"
                        />
                      ) : (
                        <p className="text-slate-500 text-sm">Image not available</p>
                      )}
                    </div>
                  </div>

                  {/* Rectified Image */}
                  {result.rectification_applied && result.rectified_image_url && (
                    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl relative overflow-hidden">
                       <div className="absolute top-0 right-0 p-2">
                         <div className="text-[10px] font-bold tracking-widest text-indigo-400/80 uppercase bg-indigo-900/30 px-2 py-1 rounded border border-indigo-500/20">DDRNet Output</div>
                       </div>
                      <h3 className="text-sm font-semibold text-slate-300 mb-4 flex items-center gap-2">
                        <Layers className="w-4 h-4 text-indigo-400" />
                        Rectified Fingerprint
                      </h3>
                      <div className="bg-slate-950 rounded-xl border border-slate-800 flex items-center justify-center p-4 h-64 overflow-hidden">
                        <img 
                          src={`http://127.0.0.1:8000${result.rectified_image_url}`} 
                          alt="Rectified Output" 
                          className="max-w-full max-h-full object-contain filter drop-shadow-md"
                        />
                      </div>
                    </div>
                  )}
                </div>

                {/* Matching Details */}
                <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
                  <h3 className="text-sm font-semibold text-slate-100 mb-6 flex items-center gap-2 uppercase tracking-wider">
                    <Search className="w-4 h-4 text-slate-400" />
                    Bozorth3 Matching Details
                  </h3>
                  
                  <div className="space-y-4">
                    <div className="flex items-center justify-between px-4 py-2 border-b border-slate-800 text-xs font-medium text-slate-500 uppercase tracking-wider">
                      <span>Enrolled Impression</span>
                      <span>Score</span>
                    </div>
                    
                    <div className="space-y-2">
                      {result.scores.map((score, idx) => {
                        const isBest = score.impression_id === result.best_impression_id;
                        return (
                          <div 
                            key={idx} 
                            className={cn(
                              "flex items-center justify-between px-4 py-3 rounded-xl border transition-colors",
                              isBest 
                                ? "bg-blue-900/20 border-blue-500/30 shadow-sm" 
                                : "bg-slate-950/50 border-slate-800/50"
                            )}
                          >
                            <div className="flex items-center gap-3">
                              {isBest && (
                                <div className="w-1.5 h-1.5 rounded-full bg-blue-500 shadow-[0_0_8px_rgba(59,130,246,0.8)]"></div>
                              )}
                              <span className={cn(
                                "text-sm font-medium",
                                isBest ? "text-blue-300" : "text-slate-300"
                              )}>
                                Impression {score.impression_id}
                              </span>
                            </div>
                            <div className="flex items-center gap-4">
                              <span className={cn(
                                "text-sm font-mono",
                                isBest ? "text-blue-400 font-bold" : "text-slate-400"
                              )}>
                                {score.score.toFixed(4)}
                              </span>
                              {isBest && (
                                <span className="text-[10px] uppercase font-bold tracking-wider text-blue-400 bg-blue-500/10 px-2 py-0.5 rounded border border-blue-500/20">
                                  Best Match
                                </span>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>

              </>
            ) : (
              // Empty State
              <div className="h-full min-h-[400px] border-2 border-dashed border-slate-800 rounded-2xl flex flex-col items-center justify-center text-center p-8 bg-slate-900/30">
                <ShieldCheck className="w-16 h-16 text-slate-700 mb-4" />
                <h3 className="text-lg font-medium text-slate-400 mb-2">Ready for Authentication</h3>
                <p className="text-sm text-slate-500 max-w-sm">
                  Upload a fingerprint image and provide an identity code to begin the biometric authentication process.
                </p>
              </div>
            )}
            
          </div>
          
        </div>
      </main>
    </div>
  );
}
