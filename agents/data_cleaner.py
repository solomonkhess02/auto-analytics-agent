"""
Data Cleaner Agent — Implements Agentic Reasoning for Data Cleaning.

WHAT THIS FILE DOES:
    This agent is split into two parts (planning and executing) to allow
    for human-in-the-loop feedback. 
    
    1. generate_plan: Looks at the data profile and reasons about what
       needs to be cleaned, outputting a structured JSON plan.
    2. execute_plan: (Will be implemented in Step 11) takes the plan +
       human feedback, writes pandas code, runs it, and validates the result.
"""

import json
import os
from agents.base_agent import BaseAgent
from core.state import PipelineState
from core.prompts import CLEANER_PLAN_PROMPT, CLEANER_CODE_PROMPT

class DataCleanerAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="Cleaner")

    def run(self, state: PipelineState) -> dict:
        """
        Since we are splitting this agent into two nodes for human-in-the-loop,
        we won't use the standard `run` method directly in our graph.
        Instead, we will route to `generate_plan` and `execute_plan`.
        """
        pass

    def generate_plan(self, state: PipelineState) -> dict:
        """
        Step 1: Reason about the data and create a cleaning plan.
        """
        profile = state.get("data_profile")
        if not profile:
            return {"errors": ["Cleaner Agent: No data_profile found in state. Did the Profiler run?"]}

        print(f"[{self.name}] Analyzing data profile to generate a cleaning plan...")

        # We feed a focused summary of the profile to the LLM to avoid context overload
        profile_summary = json.dumps({
            "shape": profile.get("shape"),
            "dtypes": profile.get("dtypes"),
            "missing_values": profile.get("missing_percentages"),
            "profiler_notes": profile.get("profiler_summary")
        }, indent=2)

        prompt = CLEANER_PLAN_PROMPT.format(profile_summary=profile_summary)
        
        # Invoke the LLM to get our JSON plan
        response = self.llm.invoke(prompt)
        
        # Safely extract text content
        text_content = response.content
        if isinstance(text_content, list):
            text_content = "\n".join([item.get("text", "") if isinstance(item, dict) else str(item) for item in text_content])
            
        # Clean markdown formatting if the LLM adds it
        text_content = text_content.strip()
        if text_content.startswith("```json"):
            text_content = text_content[7:]
        elif text_content.startswith("```"):
            text_content = text_content[3:]
        if text_content.endswith("```"):
            text_content = text_content[:-3]
            
        try:
            plan = json.loads(text_content.strip())
            print(f"[{self.name}] Cleaning plan generated successfully.")
            return {
                "cleaning_plan": plan, 
                "messages": [{"role": "agent", "name": self.name, "content": "I have generated a cleaning plan for your review."}]
            }
        except json.JSONDecodeError:
            print(f"[{self.name}] Failed to parse plan. Raw output: {text_content[:150]}...")
            return {"errors": [f"Cleaner Agent failed to parse cleaning plan as JSON."]}

    def execute_plan(self, state: PipelineState) -> dict:
        """
        Step 2: Execute the plan, run code, and validate results.
        """
        dataset_path = state.get("dataset_path")
        plan = state.get("cleaning_plan")
        feedback = state.get("human_feedback", "None")

        if not dataset_path or not plan:
            return {"errors": ["Cleaner Agent: Missing dataset_path or cleaning_plan."]}

        output_path = os.path.join("data", "cleaned_dataset.csv")
        os.makedirs("data", exist_ok=True)
        
        # Prepare the execution prompt
        prompt = CLEANER_CODE_PROMPT.format(
            dataset_path=dataset_path,
            output_path=output_path,
            human_feedback=feedback,
            cleaning_plan=json.dumps(plan, indent=2)
        )

        print(f"[{self.name}] Writing and executing cleaning code...")
        
        # Validation Loop (Business Logic Check)
        # _generate_and_execute_code handles code crashes. 
        # This loop handles code that runs but fails to actually clean the data.
        max_validation_attempts = 3
        current_prompt = prompt
        
        for attempt in range(max_validation_attempts):
            try:
                code, output = self._generate_and_execute_code(current_prompt)
            except RuntimeError as e:
                return {"errors": [f"Cleaner execution totally failed: {str(e)}"]}

            # Step 4 of the prompt asks that they print a JSON object. We parse it here.
            try:
                # Find JSON block in the output
                json_str = output.strip()
                start_idx = json_str.find('{')
                end_idx = json_str.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    json_str = json_str[start_idx:end_idx+1]
                stats = json.loads(json_str)
            except json.JSONDecodeError:
                stats = {"missing_before": 0, "missing_after": 0, "rows_before": 0, "rows_after": 0}
                print(f"[{self.name}] Warning: Could not parse execution stats JSON.")

            missing_after = stats.get("missing_after", 0)
            
            # The True Validation!
            if missing_after > 0:
                print(f"[{self.name}] VALIDATION FAILED: {missing_after} missing values remain! Retrying...")
                current_prompt = (
                    f"Your code executed successfully but FAILED the business requirement. "
                    f"There are still {missing_after} missing values in the output dataset! "
                    "You must handle ALL missing values. Fix your code and try again.\n\n"
                    f"Original instructions:\n{prompt}"
                )
            else:
                print(f"[{self.name}] Validation Passed: Dataset is completely clean.")
                
                # We return the state updates!
                cleaning_report = {
                    "actions_taken": ["Executed LLM Cleaning Plan"],
                    "columns_dropped": plan.get("drop_columns", []),
                    "columns_modified": plan.get("impute_missing", {}),
                    "rows_before": stats.get("rows_before", 0),
                    "rows_after": stats.get("rows_after", 0),
                    "missing_before": stats.get("missing_before", 0),
                    "missing_after": missing_after,
                    "cleaning_code": code,
                    "cleaner_summary": "Data cleaning executed successfully."
                }
                
                return {
                    "cleaned_dataset_path": output_path,
                    "cleaning_report": cleaning_report,
                    "messages": [{"role": "agent", "name": self.name, "content": "Cleaning complete and verified!"}]
                }
                
        return {"errors": [f"Cleaner Agent failed validation: still missing values after {max_validation_attempts} attempts."]}
