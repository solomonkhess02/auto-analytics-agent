"use client";

import { useState } from "react";
import StepIndicator from "./components/StepIndicator";
import PlanReviewCard from "./components/PlanReviewCard";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
interface DataProfile {
  shape: [number, number];
  columns: string[];
  missing_values: Record<string, number>;
  dtypes: Record<string, string>;
  profiler_summary: string;
}

interface CleaningReport {
  rows_before: number;
  rows_after: number;
  missing_before: number;
  missing_after: number;
  columns_dropped: string[];
  cleaner_summary: string;
}

interface FeatureReport {
  features_created: string[];
  features_dropped: string[];
  encoding_applied: Record<string, string>;
  scaling_applied: Record<string, string>;
  engineer_summary: string;
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function StatCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 flex flex-col gap-1">
      <div className="text-slate-500 text-xs font-semibold uppercase tracking-widest">{label}</div>
      <div className="text-slate-100 text-xl font-bold font-mono">{value}</div>
    </div>
  );
}

function SectionHeader({ icon, title, subtitle }: { icon: string; title: string; subtitle?: string }) {
  return (
    <div className="flex items-center gap-3 mb-4">
      <div className="w-10 h-10 rounded-xl bg-slate-800 border border-slate-700 flex items-center justify-center text-xl">
        {icon}
      </div>
      <div>
        <h2 className="text-lg font-bold text-slate-100">{title}</h2>
        {subtitle && <p className="text-slate-400 text-sm">{subtitle}</p>}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Page
// ---------------------------------------------------------------------------

export default function Home() {
  // Stage: 0=upload, 1=review cleaning plan, 2=review feature plan, 3=done
  const [stage, setStage] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // User inputs
  const [datasetPath, setDatasetPath] = useState("data/dummy_dirty.csv");

  // Session
  const [threadId, setThreadId] = useState<string | null>(null);

  // Data from backend
  const [dataProfile, setDataProfile] = useState<DataProfile | null>(null);
  const [cleaningPlan, setCleaningPlan] = useState<Record<string, any> | null>(null);
  const [cleaningReport, setCleaningReport] = useState<CleaningReport | null>(null);
  const [featurePlan, setFeaturePlan] = useState<Record<string, any> | null>(null);
  const [featureReport, setFeatureReport] = useState<FeatureReport | null>(null);
  const [engineeredPath, setEngineeredPath] = useState<string | null>(null);

  const apiBase = "http://localhost:8000";

  // -------------------------------------------------------------------------
  // Step 1: Start Pipeline
  // -------------------------------------------------------------------------
  const startPipeline = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await fetch(`${apiBase}/api/pipeline/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset_path: datasetPath, task_type: "auto" }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Backend error");

      setThreadId(data.thread_id);
      setDataProfile(data.data_profile);
      setCleaningPlan(data.cleaning_plan);
      setStage(1);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setIsLoading(false);
    }
  };

  // -------------------------------------------------------------------------
  // Step 2: Approve Cleaning Plan
  // -------------------------------------------------------------------------
  const approveCleaning = async (feedback: string) => {
    if (!threadId) return;
    setIsLoading(true);
    setError(null);
    try {
      const res = await fetch(`${apiBase}/api/pipeline/${threadId}/approve-cleaning`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ human_feedback: feedback }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Backend error");

      setCleaningReport(data.cleaning_report);
      setFeaturePlan(data.feature_plan);
      setStage(2);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setIsLoading(false);
    }
  };

  // -------------------------------------------------------------------------
  // Step 3: Approve Feature Plan
  // -------------------------------------------------------------------------
  const approveFeatures = async (feedback: string) => {
    if (!threadId) return;
    setIsLoading(true);
    setError(null);
    try {
      const res = await fetch(`${apiBase}/api/pipeline/${threadId}/approve-features`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ human_feedback: feedback }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Backend error");

      setFeatureReport(data.feature_report);
      setEngineeredPath(data.engineered_dataset_path);
      setStage(3);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setIsLoading(false);
    }
  };

  // -------------------------------------------------------------------------
  // Reset
  // -------------------------------------------------------------------------
  const reset = () => {
    setStage(0);
    setError(null);
    setThreadId(null);
    setDataProfile(null);
    setCleaningPlan(null);
    setCleaningReport(null);
    setFeaturePlan(null);
    setFeatureReport(null);
    setEngineeredPath(null);
  };

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------
  return (
    <main className="min-h-screen bg-slate-950 text-slate-200 p-6 md:p-10">
      <div className="max-w-3xl mx-auto space-y-8">

        {/* Header */}
        <div className="text-center space-y-3 pt-4">
          <h1 className="text-4xl md:text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-purple-400 to-emerald-400">
            Auto-Analytics Agent
          </h1>
          <p className="text-slate-400 text-base md:text-lg">
            Your autonomous AI data scientist — with human oversight at every step.
          </p>
        </div>

        {/* Step Indicator */}
        <StepIndicator currentStage={stage} />

        {/* Error Banner */}
        {error && (
          <div className="bg-red-950/60 border border-red-800 text-red-200 p-4 rounded-xl flex items-start gap-3 animate-in fade-in duration-300">
            <span className="text-xl mt-0.5">⚠️</span>
            <div>
              <p className="font-semibold">Something went wrong</p>
              <p className="text-sm text-red-300 mt-0.5">{error}</p>
            </div>
            <button onClick={() => setError(null)} className="ml-auto text-red-400 hover:text-red-200">✕</button>
          </div>
        )}

        {/* ---- STAGE 0: Upload ---- */}
        {stage === 0 && (
          <div className="bg-slate-900 border border-slate-800 p-8 rounded-2xl shadow-xl space-y-6 animate-in fade-in duration-400">
            <SectionHeader icon="📂" title="Upload Dataset" subtitle="Provide the path to your CSV file on the server." />

            <div>
              <label className="block text-sm font-medium text-slate-400 mb-2">Dataset Path</label>
              <input
                type="text"
                value={datasetPath}
                onChange={(e) => setDatasetPath(e.target.value)}
                className="w-full bg-slate-950 border border-slate-700 rounded-xl px-4 py-3 text-slate-200 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-colors font-mono text-sm"
                placeholder="data/your-dataset.csv"
              />
              <p className="text-slate-600 text-xs mt-2">Path is relative to the backend project root.</p>
            </div>

            <button
              onClick={startPipeline}
              disabled={isLoading || !datasetPath.trim()}
              className="w-full py-4 rounded-xl font-bold text-lg transition-all duration-200 shadow-lg active:scale-[0.98]
                disabled:bg-slate-800 disabled:text-slate-500 disabled:cursor-not-allowed
                bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-500 hover:to-blue-400 hover:shadow-blue-500/30 text-white flex items-center justify-center gap-3"
            >
              {isLoading ? (
                <>
                  <div className="w-5 h-5 border-2 border-white/20 border-t-white rounded-full animate-spin" />
                  Profiling Dataset...
                </>
              ) : (
                "🚀 Start Pipeline"
              )}
            </button>
          </div>
        )}

        {/* ---- STAGE 1: Data Profile + Cleaning Plan ---- */}
        {stage >= 1 && dataProfile && (
          <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
            {/* Data Profile */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-xl">
              <div className="px-6 py-4 bg-slate-800/60 border-b border-slate-700">
                <SectionHeader icon="🔍" title="Data Profile" subtitle="Here's what the Profiler Agent found." />
              </div>
              <div className="p-6 space-y-5">
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                  <StatCard label="Rows" value={dataProfile.shape[0].toLocaleString()} />
                  <StatCard label="Columns" value={dataProfile.shape[1]} />
                  <StatCard label="Missing Cells" value={Object.values(dataProfile.missing_values).reduce((a, b) => a + b, 0)} />
                </div>

                {/* Column types */}
                <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800">
                  <p className="text-xs text-slate-500 font-semibold uppercase tracking-widest mb-3">Column Types</p>
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(dataProfile.dtypes).map(([col, dtype]) => (
                      <span key={col} className="px-2 py-1 rounded-lg bg-slate-800 border border-slate-700 text-xs font-mono">
                        <span className="text-slate-300">{col}</span>
                        <span className="text-slate-500 ml-1">({dtype})</span>
                      </span>
                    ))}
                  </div>
                </div>

                {/* Missing values */}
                {Object.values(dataProfile.missing_values).some(v => v > 0) && (
                  <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800">
                    <p className="text-xs text-slate-500 font-semibold uppercase tracking-widest mb-3">Missing Values</p>
                    <div className="flex flex-wrap gap-2">
                      {Object.entries(dataProfile.missing_values).filter(([, v]) => v > 0).map(([col, count]) => (
                        <span key={col} className="px-3 py-1 rounded-md bg-amber-500/10 border border-amber-500/20 text-amber-300 text-xs font-mono flex items-center gap-1.5">
                          {col} <span className="bg-amber-500/20 px-1.5 rounded">{count}</span>
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* LLM Summary */}
                <div className="bg-gradient-to-br from-indigo-950/50 to-slate-950 p-5 rounded-xl border border-indigo-900/40">
                  <p className="text-indigo-400 text-xs font-semibold tracking-widest uppercase mb-2">✨ Agent Analysis</p>
                  <p className="text-slate-300 leading-relaxed">{dataProfile.profiler_summary}</p>
                </div>
              </div>
            </div>

            {/* Cleaning Plan — only show if still awaiting approval */}
            {stage === 1 && cleaningPlan && (
              <PlanReviewCard
                title="Cleaning Plan"
                subtitle="The Cleaner Agent proposes these actions. Review before it modifies your dataset."
                icon="🧹"
                plan={cleaningPlan}
                onApprove={approveCleaning}
                isLoading={isLoading}
              />
            )}
          </div>
        )}

        {/* ---- STAGE 2: Cleaning Report + Feature Plan ---- */}
        {stage >= 2 && cleaningReport && (
          <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
            {/* Cleaning Report */}
            <div className="bg-slate-900 border border-emerald-900/40 rounded-2xl overflow-hidden shadow-xl">
              <div className="px-6 py-4 bg-emerald-950/30 border-b border-emerald-900/30">
                <SectionHeader icon="✅" title="Cleaning Complete" subtitle="Here's what the Cleaner Agent did." />
              </div>
              <div className="p-6 space-y-4">
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  <StatCard label="Rows Before" value={cleaningReport.rows_before.toLocaleString()} />
                  <StatCard label="Rows After" value={cleaningReport.rows_after.toLocaleString()} />
                  <StatCard label="Missing Before" value={cleaningReport.missing_before} />
                  <StatCard label="Missing After" value={cleaningReport.missing_after} />
                </div>
                {cleaningReport.columns_dropped.length > 0 && (
                  <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800">
                    <p className="text-xs text-slate-500 uppercase tracking-widest mb-2">Columns Dropped</p>
                    <div className="flex flex-wrap gap-1.5">
                      {cleaningReport.columns_dropped.map((col) => (
                        <span key={col} className="px-2 py-0.5 rounded bg-red-500/10 border border-red-500/20 text-red-300 text-xs font-mono">{col}</span>
                      ))}
                    </div>
                  </div>
                )}
                <div className="bg-gradient-to-br from-emerald-950/30 to-slate-950 p-5 rounded-xl border border-emerald-900/30">
                  <p className="text-emerald-400 text-xs font-semibold tracking-widest uppercase mb-2">✨ Agent Summary</p>
                  <p className="text-slate-300 leading-relaxed text-sm">{cleaningReport.cleaner_summary}</p>
                </div>
              </div>
            </div>

            {/* Feature Plan — only show if still awaiting approval */}
            {stage === 2 && featurePlan && (
              <PlanReviewCard
                title="Feature Engineering Plan"
                subtitle="The Engineer Agent proposes these transformations. Review before it modifies your features."
                icon="⚙️"
                plan={featurePlan}
                onApprove={approveFeatures}
                isLoading={isLoading}
              />
            )}
          </div>
        )}

        {/* ---- STAGE 3: Feature Engineering Complete ---- */}
        {stage === 3 && featureReport && (
          <div className="bg-slate-900 border border-purple-900/40 rounded-2xl overflow-hidden shadow-2xl animate-in fade-in slide-in-from-bottom-4 duration-500">
            <div className="px-6 py-4 bg-purple-950/30 border-b border-purple-900/30">
              <SectionHeader icon="🎉" title="Pipeline Phases Complete!" subtitle="Data is cleaned and features are engineered. Ready for model training." />
            </div>
            <div className="p-6 space-y-4">
              {featureReport.features_created.length > 0 && (
                <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800">
                  <p className="text-xs text-slate-500 uppercase tracking-widest mb-2">New Features Created</p>
                  <div className="flex flex-wrap gap-1.5">
                    {featureReport.features_created.map((f) => (
                      <span key={f} className="px-2 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/20 text-emerald-300 text-xs font-mono">{f}</span>
                    ))}
                  </div>
                </div>
              )}
              {Object.keys(featureReport.encoding_applied).length > 0 && (
                <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800">
                  <p className="text-xs text-slate-500 uppercase tracking-widest mb-2">Encoding Applied</p>
                  <div className="space-y-1">
                    {Object.entries(featureReport.encoding_applied).map(([col, enc]) => (
                      <div key={col} className="text-xs font-mono">
                        <span className="text-slate-300">{col}</span>
                        <span className="text-slate-500"> → </span>
                        <span className="text-purple-300">{enc}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              <div className="bg-gradient-to-br from-purple-950/30 to-slate-950 p-5 rounded-xl border border-purple-900/30">
                <p className="text-purple-400 text-xs font-semibold tracking-widest uppercase mb-2">✨ Agent Summary</p>
                <p className="text-slate-300 leading-relaxed text-sm">{featureReport.engineer_summary}</p>
              </div>
              {engineeredPath && (
                <div className="bg-slate-950 border border-slate-800 p-3 rounded-lg flex items-center gap-3">
                  <span className="text-emerald-400">💾</span>
                  <div>
                    <p className="text-xs text-slate-500">Engineered dataset saved to</p>
                    <p className="text-slate-300 text-sm font-mono">{engineeredPath}</p>
                  </div>
                </div>
              )}
              <button
                onClick={reset}
                className="w-full py-3 rounded-xl font-semibold text-slate-400 border border-slate-700 hover:bg-slate-800 hover:text-slate-200 transition-all mt-2"
              >
                ↩ Run Another Dataset
              </button>
            </div>
          </div>
        )}

      </div>
    </main>
  );
}
