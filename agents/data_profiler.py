"""
Data Profiler Agent.
"""

import json
from agents.base_agent import BaseAgent
from core.state import PipelineState, DataProfile
from core.prompts import PROFILER_CODE_PROMPT, PROFILER_SUMMARY_PROMPT

class DataProfilerAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="Profiler")

    def run(self, state: PipelineState) -> dict:
        dataset_path = state.get("dataset_path")
        
        if not dataset_path:
            return {"errors": ["Profiler Agent: No dataset_path found in state."]}

        print(f"[{self.name}] Starting analysis on {dataset_path}...")

        code_prompt = PROFILER_CODE_PROMPT.format(dataset_path=dataset_path)
        try:
            code, output = self._generate_and_execute_code(code_prompt)
            print(f"[{self.name}] Profiling code executed successfully.")
        except RuntimeError as e:
            return {"errors": [f"Profiler Agent code generation failed: {str(e)}"]}

        try:
            json_str = output.strip()
            start_idx = json_str.find('{')
            end_idx = json_str.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                json_str = json_str[start_idx:end_idx+1]
                
            stats = json.loads(json_str)
        except json.JSONDecodeError as e:
            return {"errors": [f"Profiler Agent failed to parse output as JSON. Output was: {output}"][:200]}

        print(f"[{self.name}] Generating natural language summary...")
        summary_prompt = PROFILER_SUMMARY_PROMPT.format(
            shape=stats.get('shape'),
            missing_values=stats.get('missing_values'),
            dtypes=stats.get('dtypes')
        )
        summary_response = self.llm.invoke(summary_prompt)
        
        content = summary_response.content
        if isinstance(content, list):
            summary_text = "\n".join([item.get("text", "") if isinstance(item, dict) else str(item) for item in content])
        else:
            summary_text = str(content)
            
        profile: DataProfile = {
            "shape": tuple(stats.get("shape", [0, 0])),
            "columns": stats.get("columns", []),
            "dtypes": stats.get("dtypes", {}),
            "missing_values": stats.get("missing_values", {}),
            "missing_percentages": stats.get("missing_percentages", {}),
            "descriptive_stats": {}, 
            "unique_counts": stats.get("unique_counts", {}),
            "sample_rows": [], 
            "correlations": {}, 
            "target_column": None,
            "task_type": None,
            "profiler_summary": summary_text.strip()
        }

        print(f"[{self.name}] Profiling complete!")
        return {
            "data_profile": profile,
            "messages": [{"role": "agent", "name": self.name, "content": "Dataset profiling complete."}]
        }
