"use client";

import { useState } from "react";

export default function Home() {
  const [datasetPath, setDatasetPath] = useState("data/dummy.csv");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const runPipeline = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      // Send a request to our FastAPI backend!
      const response = await fetch("http://localhost:8000/api/run", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          dataset_path: datasetPath,
          task_type: "auto",
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Something went wrong in the backend.");
      }

      setResult(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-slate-950 p-8 text-slate-200">
      <div className="max-w-4xl mx-auto space-y-8">
        
        {/* Header */}
        <div className="text-center space-y-4">
          <h1 className="text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-emerald-400">
            Auto-Analytics Agent
          </h1>
          <p className="text-slate-400 text-lg">
            Your autonomous AI data scientist.
          </p>
        </div>

        {/* Control Panel */}
        <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-xl flex flex-col md:flex-row gap-4 items-center">
          <div className="flex-1 w-full">
            <label className="block text-sm font-medium text-slate-400 mb-1">
              Dataset Path
            </label>
            <input
              type="text"
              value={datasetPath}
              onChange={(e) => setDatasetPath(e.target.value)}
              className="w-full bg-slate-950 border border-slate-700 rounded-lg px-4 py-3 text-slate-200 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-colors"
              placeholder="e.g. data/dummy.csv"
            />
          </div>
          <button
            onClick={runPipeline}
            disabled={loading}
            className={`mt-6 px-8 py-3 rounded-lg font-bold transition-all duration-200 shadow-lg
              ${loading 
                ? 'bg-slate-700 text-slate-400 cursor-not-allowed' 
                : 'bg-blue-600 hover:bg-blue-500 hover:shadow-blue-500/25 text-white active:scale-95'
              }`}
          >
            {loading ? (
              <span className="flex items-center gap-2">
                <div className="w-4 h-4 border-2 border-white/20 border-t-white rounded-full animate-spin" />
                Analyzing Agents...
              </span>
            ) : "🚀 Run Pipeline"}
          </button>
        </div>

        {/* Error State */}
        {error && (
          <div className="bg-red-950/50 border border-red-900 text-red-200 p-4 rounded-xl flex items-start gap-3">
            <span className="text-xl">⚠️</span>
            <p>{error}</p>
          </div>
        )}

        {/* Results State */}
        {result && result.status === "success" && (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-2xl animate-in fade-in slide-in-from-bottom-4 duration-500">
            <div className="bg-slate-800 px-6 py-4 border-b border-slate-700">
              <h2 className="text-xl font-semibold flex items-center gap-2">
                📊 Phase Complete: <span className="text-emerald-400 uppercase tracking-widest text-sm bg-emerald-400/10 px-3 py-1 rounded-full">{result.phase}</span>
              </h2>
            </div>
            
            <div className="p-6 space-y-6">
              
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
                  <div className="text-slate-500 text-sm font-medium mb-1">Dataset Shape</div>
                  <div className="text-2xl font-mono text-slate-200">
                    {result.data_profile.shape[0]} <span className="text-slate-600 text-sm">rows</span> × {result.data_profile.shape[1]} <span className="text-slate-600 text-sm">cols</span>
                  </div>
                </div>

                <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
                  <div className="text-slate-500 text-sm font-medium mb-1">Columns Tracked</div>
                  <div className="text-sm font-mono text-blue-300 break-words leading-relaxed">
                    {result.data_profile.columns.join(", ")}
                  </div>
                </div>
              </div>

              <div className="bg-slate-950 p-5 rounded-xl border border-slate-800 space-y-3">
                <div className="text-slate-500 text-sm font-medium">Missing Values Detected</div>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(result.data_profile.missing_values).map(([col, count]: any) => (
                    <div key={col} className={`px-3 py-1 rounded-md text-sm font-mono flex items-center gap-2 ${count > 0 ? 'bg-amber-500/10 text-amber-300 border border-amber-500/20' : 'bg-slate-800 text-slate-400'}`}>
                      <span>{col}</span>
                      <span className="bg-black/20 px-2 py-0.5 rounded text-xs">{count}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="bg-gradient-to-br from-indigo-950/50 to-slate-950 p-6 rounded-xl border border-indigo-900/50 relative overflow-hidden">
                <div className="absolute top-0 right-0 p-4 opacity-10">✨</div>
                <div className="text-indigo-300 text-sm font-semibold tracking-wider mb-2 uppercase">Agent Analysis</div>
                <p className="text-slate-300 leading-relaxed text-lg">
                  {result.data_profile.profiler_summary}
                </p>
              </div>

            </div>
          </div>
        )}
      </div>
    </main>
  );
}
