"use client";

import { useState } from "react";

// ============================================================================
// PlanReviewCard — Displays a cleaning or feature plan for human approval
// ============================================================================

interface PlanReviewCardProps {
  title: string;
  subtitle: string;
  icon: string;
  plan: Record<string, any>;
  onApprove: (feedback: string) => void;
  isLoading: boolean;
}

// Renders a single JSON value prettily — handles arrays, objects, strings
function PlanValueRenderer({ value }: { value: any }) {
  if (Array.isArray(value)) {
    if (value.length === 0)
      return <span className="text-slate-500 italic">none</span>;
    return (
      <div className="flex flex-wrap gap-1.5 mt-1">
        {value.map((item, i) => (
          <span
            key={i}
            className="px-2 py-0.5 rounded-md bg-blue-500/10 border border-blue-500/20 text-blue-300 text-xs font-mono"
          >
            {String(item)}
          </span>
        ))}
      </div>
    );
  }

  if (typeof value === "object" && value !== null) {
    return (
      <div className="mt-1 space-y-1 pl-3 border-l border-slate-700">
        {Object.entries(value).map(([k, v]) => (
          <div key={k} className="text-xs">
            <span className="text-slate-400 font-mono">{k}: </span>
            <span className="text-amber-300 font-mono">{String(v)}</span>
          </div>
        ))}
      </div>
    );
  }

  return (
    <span className="text-slate-300 font-mono text-sm">{String(value)}</span>
  );
}

export default function PlanReviewCard({
  title,
  subtitle,
  icon,
  plan,
  onApprove,
  isLoading,
}: PlanReviewCardProps) {
  const [feedback, setFeedback] = useState("");
  const [isModifying, setIsModifying] = useState(false);

  const handleApprove = () => {
    onApprove("Looks good, proceed.");
  };

  const handleModify = () => {
    if (!isModifying) {
      setIsModifying(true);
      return;
    }
    if (!feedback.trim()) return;
    onApprove(feedback);
  };

  // Map plan keys to friendly display names
  const keyLabels: Record<string, string> = {
    drop_columns: "Columns to Drop",
    impute_missing: "Missing Value Strategy",
    type_conversions: "Type Corrections",
    outlier_handling: "Outlier Handling",
    handle_duplicates: "Duplicate Handling",
    reasoning: "Agent Reasoning",
    datetime_features: "Datetime Features to Extract",
    categorical_encoding: "Categorical Encoding",
    numerical_scaling: "Numerical Scaling",
    new_features: "New Features to Create",
  };

  return (
    <div className="bg-slate-900 border border-slate-700 rounded-2xl overflow-hidden shadow-2xl animate-in fade-in slide-in-from-bottom-4 duration-500">
      {/* Header */}
      <div className="bg-gradient-to-r from-slate-800 to-slate-900 px-6 py-4 border-b border-slate-700 flex items-start justify-between">
        <div>
          <h2 className="text-xl font-bold text-slate-100 flex items-center gap-2">
            <span>{icon}</span> {title}
          </h2>
          <p className="text-slate-400 text-sm mt-0.5">{subtitle}</p>
        </div>
        <span className="text-xs px-3 py-1 rounded-full bg-amber-500/15 border border-amber-500/30 text-amber-300 font-semibold tracking-wider">
          AWAITING APPROVAL
        </span>
      </div>

      {/* Plan body */}
      <div className="p-6 space-y-4">
        {Object.entries(plan).map(([key, value]) => (
          <div key={key} className="bg-slate-950/60 rounded-xl p-4 border border-slate-800">
            <div className="text-xs font-semibold tracking-widest uppercase text-slate-500 mb-2">
              {keyLabels[key] || key.replace(/_/g, " ")}
            </div>
            <PlanValueRenderer value={value} />
          </div>
        ))}
      </div>

      {/* Modify text area */}
      {isModifying && (
        <div className="px-6 pb-4 animate-in slide-in-from-top-2 duration-300">
          <label className="text-sm font-medium text-slate-400 mb-1 block">
            Your Instructions to the Agent
          </label>
          <textarea
            value={feedback}
            onChange={(e) => setFeedback(e.target.value)}
            rows={3}
            className="w-full bg-slate-950 border border-slate-600 rounded-lg px-4 py-3 text-slate-200 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-colors text-sm font-mono resize-none"
            placeholder="e.g. Don't drop the age column, impute it with median instead."
          />
        </div>
      )}

      {/* Action buttons */}
      <div className="px-6 pb-6 flex items-center gap-3">
        <button
          onClick={handleApprove}
          disabled={isLoading || isModifying}
          className="flex-1 py-3 rounded-xl font-bold text-white bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-700 disabled:text-slate-500 disabled:cursor-not-allowed transition-all duration-200 shadow-lg hover:shadow-emerald-500/25 active:scale-95 flex items-center justify-center gap-2"
        >
          {isLoading ? (
            <>
              <div className="w-4 h-4 border-2 border-white/20 border-t-white rounded-full animate-spin" />
              Running Agent...
            </>
          ) : (
            "✅ Approve & Run"
          )}
        </button>

        <button
          onClick={handleModify}
          disabled={isLoading}
          className={`py-3 px-6 rounded-xl font-bold transition-all duration-200 active:scale-95 border
            ${isModifying && feedback.trim()
              ? "bg-blue-600 hover:bg-blue-500 text-white border-blue-500 shadow-lg hover:shadow-blue-500/25"
              : "bg-slate-800 hover:bg-slate-700 text-slate-300 border-slate-700"
            }
          `}
        >
          {isModifying ? (feedback.trim() ? "✏️ Submit Changes" : "Type above...") : "✏️ Modify"}
        </button>

        {isModifying && (
          <button
            onClick={() => { setIsModifying(false); setFeedback(""); }}
            className="py-3 px-4 rounded-xl text-slate-400 hover:text-slate-200 transition-colors text-sm"
          >
            Cancel
          </button>
        )}
      </div>
    </div>
  );
}
