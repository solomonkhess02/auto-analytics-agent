"""
Model Evaluator Agent.
"""

import json
import os
import re
from agents.base_agent import BaseAgent
from core.state import PipelineState, EvaluationReport
from core.prompts import MODEL_EVALUATION_PROMPT, EVALUATION_SUMMARY_PROMPT


class EvaluatorAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="Evaluator")

    def run(self, state: PipelineState) -> dict:
        training_results = state.get("training_results")
        artifacts_dir = state.get("model_artifacts_dir") or "models"
        target_column = state.get("target_column")
        task_type = state.get("task_type")

        if not training_results:
            return {"errors": ["Evaluator Agent: No training_results found in state."]}

        # Identify the best model based on cross_val_mean
        best_model = max(training_results, key=lambda x: x.get("cross_val_mean", 0.0))
        best_model_name = best_model["model_name"]

        # Prefer the exact path the Trainer reported it saved the model to.
        # Fall back to reconstructing the filename only if it wasn't provided.
        best_model_file = best_model.get("model_file")
        if not best_model_file:
            model_name_sanitized = re.sub(r'[^a-zA-Z0-9]', '_', best_model_name).lower()
            model_name_sanitized = re.sub(r'_+', '_', model_name_sanitized).strip('_')
            best_model_file = os.path.join(artifacts_dir, f"{model_name_sanitized}.joblib")

        print(f"[{self.name}] Best model identified: '{best_model_name}' (file: {best_model_file})")

        report_dir = os.path.join("reports")
        os.makedirs(report_dir, exist_ok=True)
        report_html_path = os.path.join(report_dir, "evaluation_report.html")

        # Create evaluation prompt
        prompt = MODEL_EVALUATION_PROMPT.format(
            artifacts_dir=artifacts_dir,
            target_column=target_column,
            task_type=task_type,
            best_model_name=best_model_name,
            best_model_file=best_model_file.replace("\\", "/"),
            report_html_path=report_html_path.replace("\\", "/")
        )

        try:
            code, output = self._generate_and_execute_code(prompt)
            print(f"[{self.name}] Model evaluation code executed successfully.")
        except RuntimeError as e:
            return {"errors": [f"Evaluator Agent code generation failed: {str(e)}"]}

        try:
            json_str = output.strip()
            start_idx = json_str.find('{')
            end_idx = json_str.rfind('}')
            if start_idx != -1 and end_idx != -1:
                json_str = json_str[start_idx:end_idx+1]
            eval_data = json.loads(json_str)
        except json.JSONDecodeError:
            eval_data = {}
            print(f"[{self.name}] Warning: Could not parse evaluation results JSON. Output was: {output}")

        print(f"[{self.name}] Generating natural language evaluation summary...")
        
        # Extract top features string for summary prompt
        top_features = eval_data.get("feature_importances", {})

        def _fmt_importance(value):
            try:
                return f"{float(value):.3f}"
            except (TypeError, ValueError):
                return str(value)

        top_features_summary = ", ".join(
            [f"{k} ({_fmt_importance(v)})" for k, v in list(top_features.items())[:5]]
        )

        summary_prompt = EVALUATION_SUMMARY_PROMPT.format(
            best_model_name=best_model_name,
            task_type=task_type,
            metrics=json.dumps(eval_data.get("metrics", {}), indent=2),
            top_features=top_features_summary or "None"
        )
        
        summary_response = self.llm.invoke(summary_prompt)
        summary_content = summary_response.content
        if isinstance(summary_content, list):
            summary_text = "\n".join([item.get("text", "") if isinstance(item, dict) else str(item) for item in summary_content])
        else:
            summary_text = str(summary_content)

        # Form the evaluation report dict
        evaluation_report: EvaluationReport = {
            "best_model_name": best_model_name,
            "metrics": eval_data.get("metrics", {}),
            "confusion_matrix": eval_data.get("confusion_matrix"),
            "classification_report": eval_data.get("classification_report"),
            "feature_importances": eval_data.get("feature_importances", {}),
            "plot_paths": eval_data.get("plot_paths", []),
            "natural_language_report": summary_text.strip(),
            "report_html_path": report_html_path
        }

        print(f"[{self.name}] Evaluation phase complete.")

        return {
            "evaluation_report": evaluation_report,
            "messages": [{"role": "agent", "name": self.name, "content": "Model evaluation and HTML reporting complete."}]
        }
