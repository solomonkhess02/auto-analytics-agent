"""
Data Profiler Agent — Analyzes the raw dataset.

WHAT THIS FILE DOES:
    This is our first concrete agent. It inherits from BaseAgent.
    Its job is to look at the dataset uploaded by the user, write Python
    code to calculate statistics (shape, missing values, etc.), and generate
    a human-readable summary.

HOW IT WORKS:
    1. Reads `dataset_path` from the pipeline state.
    2. Asks the LLM to write pandas code that profiles the data and prints JSON.
    3. Uses BaseAgent._generate_and_execute_code() to run the code.
        -> If the code fails, BaseAgent handles the self-healing retries!
    4. Parses the JSON output into a Python dictionary.
    5. Asks the LLM to write a natural language summary of the findings.
    6. Returns the data to update the PipelineState.
"""

import json
from agents.base_agent import BaseAgent
from core.state import PipelineState, DataProfile

class DataProfilerAgent(BaseAgent):
    def __init__(self):
        # Call the parent class constructor and give this agent a name
        super().__init__(name="Profiler")

    def run(self, state: PipelineState) -> dict:
        """
        The main entry point for this agent.
        Takes the current state, adds the data profile, and returns the update.
        """
        dataset_path = state.get("dataset_path")
        
        if not dataset_path:
            # We can't profile if we don't know where the data is!
            return {"errors": ["Profiler Agent: No dataset_path found in state."]}

        print(f"[{self.name}] Starting analysis on {dataset_path}...")

        # --- Phase 1: Generate & Run Code ---
        # We ask the LLM to write a Python script that analyzes the data
        # and prints the results as a JSON string.
        code_prompt = f"""
You are a Data Scientist Agent. Analyze the dataset at '{dataset_path}'.

Write a Python script that:
1. Loads the dataset using pandas.
2. Calculates the following statistics:
   - shape (list of 2 integers: [rows, columns])
   - columns (list of strings)
   - dtypes (dictionary mapping column names to string representations of their types)
   - missing_values (dictionary mapping column names to missing value counts)
   - missing_percentages (dictionary mapping column names to percentage of missing values as floats)
   - unique_counts (dictionary mapping column names to number of unique values)
3. Outputs the final result EXACTLY as a valid JSON string using `json.dumps()`.
4. Ensure the output only contains the JSON string and absolutely nothing else. Print the JSON at the very end.

IMPORTANT: Use proper error handling when reading the file.
"""
        try:
            # This calls the method we wrote in BaseAgent!
            # It will automatically retry (self-heal) if the code crashes.
            code, output = self._generate_and_execute_code(code_prompt)
            print(f"[{self.name}] Profiling code executed successfully.")
            
        except RuntimeError as e:
            # The LLM failed 3 times in a row
            return {"errors": [f"Profiler Agent code generation failed: {str(e)}"]}

        # --- Phase 2: Parse Results ---
        try:
            # We expect the output to be JSON since we told the Python script to print JSON
            # However, sometimes Python prints warnings or other text. 
            # We try to extract just the JSON part.
            
            # Simple clean up just in case there are single quotes instead of double quotes, 
            # though json.dumps should output double quotes.
            json_str = output.strip()
            
            # Find the first '{' and last '}' to strip out any warnings pandas might have printed above it
            start_idx = json_str.find('{')
            end_idx = json_str.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                json_str = json_str[start_idx:end_idx+1]
                
            stats = json.loads(json_str)
            
        except json.JSONDecodeError as e:
            return {"errors": [f"Profiler Agent failed to parse output as JSON. Output was: {output}"][:200]}

        # --- Phase 3: Natural Language Summary ---
        # We now have the raw numbers. Let's ask the LLM to explain them like a human.
        print(f"[{self.name}] Generating natural language summary...")
        summary_prompt = f"""
You are a Data Science consultant. Review these dataset statistics:
Shape: {stats.get('shape')}
Missing Values: {stats.get('missing_values')}
Data Types: {stats.get('dtypes')}

Write a concise, professional 3-sentence summary of this dataset.
Mention its size, any data quality issues (like missing values), and the general composition of the columns.
"""
        summary_response = self.llm.invoke(summary_prompt)
        
        # Extract text safely (handling newest Gemini list format)
        content = summary_response.content
        if isinstance(content, list):
            summary_text = "\n".join([item.get("text", "") if isinstance(item, dict) else str(item) for item in content])
        else:
            summary_text = str(content)
            
        # --- Phase 4: Construct the state update ---
        # We package everything into our DataProfile TypedDict schema defined in core/state.py
        profile: DataProfile = {
            "shape": tuple(stats.get("shape", [0, 0])),
            "columns": stats.get("columns", []),
            "dtypes": stats.get("dtypes", {}),
            "missing_values": stats.get("missing_values", {}),
            "missing_percentages": stats.get("missing_percentages", {}),
            "descriptive_stats": {}, # Skipping detailed descriptives for this basic example to keep JSON small
            "unique_counts": stats.get("unique_counts", {}),
            "sample_rows": [], # Skipping sample rows to avoid huge JSON strings
            "correlations": {}, # Skipping correlations 
            "target_column": None, # Target column might be set later
            "task_type": None,
            "profiler_summary": summary_text.strip()
        }

        # We return a dictionary that tells LangGraph how to update the pipeline state
        print(f"[{self.name}] Profiling complete!")
        return {
            "data_profile": profile,
            "messages": [{"role": "agent", "name": self.name, "content": "Dataset profiling complete."}]
        }
