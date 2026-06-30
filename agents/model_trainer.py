"""
Model Trainer Agent.
"""

import json
import os
import pandas as pd
from agents.base_agent import BaseAgent
from core.state import PipelineState, TrainingResult
from core.prompts import MODEL_TRAINING_PROMPT


class ModelTrainerAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="Trainer")

    def run(self, state: PipelineState) -> dict:
        dataset_path = state.get("engineered_dataset_path") or state.get("cleaned_dataset_path") or state.get("dataset_path")
        target_column = state.get("target_column")
        task_type = state.get("task_type") or "auto"

        if not dataset_path:
            return {"errors": ["Trainer Agent: No dataset path found in state."]}

        # Load data to resolve target_column and task_type if they are auto/None
        try:
            df = pd.read_csv(dataset_path)
        except Exception as e:
            return {"errors": [f"Trainer Agent failed to load dataset: {str(e)}"]}

        if not target_column:
            target_column = df.columns[-1]
            print(f"[{self.name}] No target_column specified. Auto-selected last column: '{target_column}'")

        if task_type == "auto" or not task_type:
            # Simple heuristic
            unique_vals = df[target_column].nunique()
            col_type = str(df[target_column].dtype)
            if unique_vals <= 10 or col_type in ["object", "category", "bool"]:
                task_type = "classification"
            else:
                task_type = "regression"
            print(f"[{self.name}] Auto-determined task_type: '{task_type}'")

        print(f"[{self.name}] Starting model training on target '{target_column}' ({task_type})...")

        artifacts_dir = os.path.join("models")
        os.makedirs(artifacts_dir, exist_ok=True)

        prompt = MODEL_TRAINING_PROMPT.format(
            engineered_dataset_path=dataset_path,
            target_column=target_column,
            task_type=task_type,
            artifacts_dir=artifacts_dir
        )

        try:
            code, output = self._generate_and_execute_code(prompt)
            print(f"[{self.name}] Model training code executed successfully.")
        except RuntimeError as e:
            return {"errors": [f"Trainer Agent code generation failed: {str(e)}"]}

        try:
            json_str = output.strip()
            start_idx = json_str.find('[')
            end_idx = json_str.rfind(']')
            if start_idx != -1 and end_idx != -1:
                json_str = json_str[start_idx:end_idx+1]
            results = json.loads(json_str)
        except json.JSONDecodeError:
            results = []
            print(f"[{self.name}] Warning: Could not parse training results JSON. Output was: {output}")

        print(f"[{self.name}] Model training complete. Trained {len(results)} models.")

        return {
            "training_results": results,
            "model_artifacts_dir": artifacts_dir,
            "target_column": target_column,
            "task_type": task_type,
            "messages": [{"role": "agent", "name": self.name, "content": f"Model training complete. Trained models: {', '.join([r.get('model_name', '') for r in results])}"}]
        }
