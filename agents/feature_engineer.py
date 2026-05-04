"""
Feature Engineer Agent.
"""

import json
import os
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

    def select_features(self, state: PipelineState) -> dict:
        pass
