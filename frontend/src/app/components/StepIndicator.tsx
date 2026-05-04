"use client";

// ============================================================================
// StepIndicator — Shows the user where they are in the pipeline
// ============================================================================

const STEPS = [
  { id: 0, label: "Upload", icon: "📂" },
  { id: 1, label: "Profiling", icon: "🔍" },
  { id: 2, label: "Cleaning", icon: "🧹" },
  { id: 3, label: "Engineering", icon: "⚙️" },
  { id: 4, label: "Complete", icon: "✅" },
];

interface StepIndicatorProps {
  currentStage: number;
}

export default function StepIndicator({ currentStage }: StepIndicatorProps) {
  return (
    <div className="flex items-center justify-center w-full gap-0">
      {STEPS.map((step, idx) => {
        const isDone = currentStage > step.id;
        const isActive = currentStage === step.id;

        return (
          <div key={step.id} className="flex items-center">
            {/* Step node */}
            <div className="flex flex-col items-center gap-1.5">
              <div
                className={`
                  w-10 h-10 rounded-full flex items-center justify-center text-lg
                  font-bold transition-all duration-500 border-2
                  ${isDone
                    ? "bg-emerald-500 border-emerald-400 text-white shadow-lg shadow-emerald-500/30"
                    : isActive
                    ? "bg-blue-600 border-blue-400 text-white shadow-lg shadow-blue-500/40 animate-pulse"
                    : "bg-slate-800 border-slate-700 text-slate-500"
                  }
                `}
              >
                {isDone ? "✓" : step.icon}
              </div>
              <span
                className={`text-xs font-medium transition-colors duration-300 ${
                  isDone
                    ? "text-emerald-400"
                    : isActive
                    ? "text-blue-300"
                    : "text-slate-600"
                }`}
              >
                {step.label}
              </span>
            </div>

            {/* Connector line (not after last item) */}
            {idx < STEPS.length - 1 && (
              <div
                className={`h-0.5 w-12 md:w-20 mb-5 mx-1 transition-all duration-700 ${
                  currentStage > step.id ? "bg-emerald-500" : "bg-slate-700"
                }`}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}
