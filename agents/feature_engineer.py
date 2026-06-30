"""
Feature Engineer Agent.
"""

import json
import os
import pandas as pd
import numpy as np
from agents.base_agent import BaseAgent
from core.state import PipelineState
from core.prompts import (
    FEATURE_ENGINEER_PLAN_PROMPT,
    FEATURE_ENGINEER_EXECUTE_PROMPT,
    FEATURE_SELECTION_PROMPT
)

class FeatureEngineerAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="Engineer")

    def run(self, state: PipelineState) -> dict:
        pass

    def generate_feature_plan(self, state: PipelineState) -> dict:
        profile = state.get("data_profile")
        task_type = state.get("task_type")
        target_column = state.get("target_column")
        
        if not profile:
            return {"errors": ["Engineer Agent: No data_profile found in state. Did the Profiler run?"]}

        print(f"[{self.name}] Analyzing data to generate a feature engineering plan...")

        profile_summary = json.dumps({
            "shape": profile.get("shape"),
            "dtypes": profile.get("dtypes"),
            "correlations": profile.get("correlations"),
            "profiler_notes": profile.get("profiler_summary")
        }, indent=2)

        prompt = FEATURE_ENGINEER_PLAN_PROMPT.format(
            task_type=task_type,
            target_column=target_column,
            profile_summary=profile_summary
        )
        
        response = self.llm.invoke(prompt)
        
        text_content = response.content
        if isinstance(text_content, list):
            text_content = "\n".join([item.get("text", "") if isinstance(item, dict) else str(item) for item in text_content])
            
        text_content = text_content.strip()
        if text_content.startswith("```json"):
            text_content = text_content[7:]
        elif text_content.startswith("```"):
            text_content = text_content[3:]
        if text_content.endswith("```"):
            text_content = text_content[:-3]
            
        try:
            plan = json.loads(text_content.strip())
            print(f"[{self.name}] Feature engineering plan generated successfully.")
            return {
                "feature_plan": plan, 
                "messages": [{"role": "agent", "name": self.name, "content": "I have generated a feature engineering plan for your review."}]
            }
        except json.JSONDecodeError:
            print(f"[{self.name}] Failed to parse plan. Raw output: {text_content[:150]}...")
            return {"errors": [f"Engineer Agent failed to parse feature plan as JSON."]}

    def execute_plan(self, state: PipelineState) -> dict:
        dataset_path = state.get("cleaned_dataset_path")
        plan = state.get("feature_plan")
        feedback = state.get("human_feedback", "None")
        target_column = state.get("target_column", "None")

        if not dataset_path or not plan:
            return {"errors": ["Engineer Agent: Missing cleaned_dataset_path or feature_plan."]}

        output_path = os.path.join("data", "engineered_dataset.csv")
        os.makedirs("data", exist_ok=True)
        
        prompt = FEATURE_ENGINEER_EXECUTE_PROMPT.format(
            cleaned_dataset_path=dataset_path,
            engineered_dataset_path=output_path,
            target_column=target_column,
            human_feedback=feedback,
            feature_plan=json.dumps(plan, indent=2)
        )

        print(f"[{self.name}] Writing and executing feature engineering code...")
        
        try:
            code, output = self._generate_and_execute_code(prompt)
        except RuntimeError as e:
            return {"errors": [f"Engineer execution totally failed: {str(e)}"]}

        try:
            json_str = output.strip()
            start_idx = json_str.find('{')
            end_idx = json_str.rfind('}')
            if start_idx != -1 and end_idx != -1:
                json_str = json_str[start_idx:end_idx+1]
            stats = json.loads(json_str)
        except json.JSONDecodeError:
            stats = {"features_before": 0, "features_after": 0, "new_columns": [], "dropped_columns": []}
            print(f"[{self.name}] Warning: Could not parse execution stats JSON.")

        print(f"[{self.name}] Feature engineering code successfully executed.")
        
        feature_report = {
            "features_created": stats.get("new_columns", []),
            "features_dropped": stats.get("dropped_columns", []),
            "encoding_applied": plan.get("categorical_encoding", {}),
            "scaling_applied": { "method": plan.get("numerical_scaling", "") },
            "feature_importances": {},
            "final_feature_list": [],
            "feature_code": code,
            "engineer_summary": "Features successfully engineered."
        }
        
        return {
            "engineered_dataset_path": output_path,
            "feature_report": feature_report,
            "messages": [{"role": "agent", "name": self.name, "content": "Feature engineering transformations applied."}]
        }

    def _strip_json_fences(self, text) -> str:
        """Normalize an LLM response into a bare JSON string."""
        if isinstance(text, list):
            text = "\n".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in text
            )
        text = str(text).strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    def select_features(self, state: PipelineState) -> dict:
        """
        Refine the engineered feature set by dropping low-importance / redundant
        features. A RandomForest provides importance scores; the LLM makes the
        final drop decision (so domain reasoning can override a naive threshold).

        This step is best-effort: any problem yields a warning and leaves the
        dataset unchanged, so it never breaks the pipeline.
        """
        dataset_path = state.get("engineered_dataset_path")
        target_column = state.get("target_column")
        task_type = state.get("task_type") or "classification"

        if not dataset_path:
            return {"warnings": ["Engineer Agent: No engineered_dataset_path found; skipping feature selection."]}

        try:
            df = pd.read_csv(dataset_path)
        except Exception as e:
            return {"warnings": [f"Engineer Agent: could not load engineered dataset for selection: {e}"]}

        if not target_column or target_column not in df.columns:
            return {"warnings": [f"Engineer Agent: target column '{target_column}' not in engineered dataset; skipping feature selection."]}

        print(f"[{self.name}] Computing feature importances for selection...")

        # Build a numeric feature matrix (X) and the target (y).
        feature_df = df.drop(columns=[target_column])
        X = feature_df.select_dtypes(include=[np.number]).fillna(0)
        if X.shape[1] <= 2:
            return {"warnings": ["Engineer Agent: too few numeric features to select from; skipping feature selection."]}

        y = df[target_column]

        try:
            if task_type == "regression":
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                from sklearn.ensemble import RandomForestClassifier
                if y.dtype == object or str(y.dtype) == "category":
                    y = y.astype("category").cat.codes
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
        except Exception as e:
            return {"warnings": [f"Engineer Agent: importance model failed to fit; skipping feature selection: {e}"]}

        importances = {
            col: float(score)
            for col, score in sorted(
                zip(X.columns, model.feature_importances_),
                key=lambda kv: kv[1],
                reverse=True,
            )
        }

        # Ask the LLM which features to drop, given the importance scores.
        features_to_drop = []
        reasoning = ""
        try:
            prompt = FEATURE_SELECTION_PROMPT.format(
                feature_importances=json.dumps(importances, indent=2)
            )
            response = self.llm.invoke(prompt)
            decision = json.loads(self._strip_json_fences(response.content))
            features_to_drop = decision.get("features_to_drop", []) or []
            reasoning = decision.get("reasoning", "")
        except Exception as e:
            print(f"[{self.name}] Feature selection decision unavailable ({e}); keeping all features.")
            features_to_drop = []

        # Safety: only drop columns that exist, never the target, and always
        # keep at least the target plus one feature.
        features_to_drop = [c for c in features_to_drop if c in df.columns and c != target_column]
        if len(df.columns) - len(features_to_drop) <= 2:
            print(f"[{self.name}] Drop list too aggressive; keeping all features.")
            features_to_drop = []

        if features_to_drop:
            df = df.drop(columns=features_to_drop)
            df.to_csv(dataset_path, index=False)
            print(f"[{self.name}] Dropped {len(features_to_drop)} feature(s): {features_to_drop}")
        else:
            print(f"[{self.name}] No features dropped.")

        final_feature_list = [c for c in df.columns if c != target_column]

        existing_report = state.get("feature_report") or {}
        updated_report = {
            **existing_report,
            "feature_importances": importances,
            "final_feature_list": final_feature_list,
            "features_dropped": sorted(
                set(existing_report.get("features_dropped", []) + features_to_drop)
            ),
        }
        if reasoning:
            updated_report["engineer_summary"] = (
                f"{existing_report.get('engineer_summary', '').strip()} "
                f"Feature selection: {reasoning}"
            ).strip()

        return {
            "feature_report": updated_report,
            "messages": [{
                "role": "agent",
                "name": self.name,
                "content": f"Feature selection complete. Dropped {len(features_to_drop)} feature(s)."
            }]
        }
