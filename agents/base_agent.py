"""
Base Agent — The blueprint for all agents in the pipeline.

WHAT THIS FILE DOES:
    Provides an Abstract Base Class (ABC) that all other agents inherit from.
    It implements the core "Agentic Loop" (the ReAct pattern):
    
    1. Ask the LLM to generate code.
    2. Run the code in the sandbox.
    3. If it fails, ask the LLM to fix it (Self-Healing).
    4. Repeat up to max_retries times.

WHY A BASE CLASS:
    Every agent (Profiler, Cleaner, Engineer) follows this exact same pattern.
    Instead of rewriting the self-healing loop for every agent, we write it
    once here.
"""

from abc import ABC, abstractmethod
import re

from core.state import PipelineState
from core.llm import get_llm
from tools.code_executor import execute_code
from core.prompts import NO_CODE_FOUND_PROMPT, RETRY_CODE_PROMPT


class BaseAgent(ABC):
    def __init__(self, name: str):
        self.name = name
        self.llm = get_llm()           # Every agent gets an LLM connection
        self.max_retries = 3           # How many times to try self-healing

    @abstractmethod
    def run(self, state: PipelineState) -> dict:
        """
        Execute agent logic.
        Each specific agent (Profiler, Cleaner, etc.) must implement this.
        
        Args:
            state: The current pipeline state.
            
        Returns:
            A dictionary containing the state updates to apply.
        """
        pass

    def _generate_and_execute_code(self, prompt: str) -> tuple[str, str]:
        """
        The core agentic self-healing loop.
        
        Args:
            prompt: The instruction to send to the LLM.
            
        Returns:
            A tuple of (executed_code, output_of_code).
            
        Raises:
            RuntimeError: If the code fails after max_retries.
        """
        current_prompt = prompt
        
        for attempt in range(self.max_retries):
            # 1. Reason: Ask the LLM what to do
            response = self.llm.invoke(current_prompt)
            
            # 2. Extract code from the LLM's text response
            code = self._extract_code(response.content)
            
            if not code:
                # The LLM forgot to put code in a markdown block
                print(f"[{self.name}] No code found in response on attempt {attempt + 1}")
                current_prompt = NO_CODE_FOUND_PROMPT.format(prompt=prompt)
                continue
                
            # 3. Act: Run the code safely in the sandbox
            success, output = execute_code(code)
            
            # 4. Observe: Did it work?
            if success:
                # Yay! It worked.
                return code, output
                
            # 5. Retry: It failed. Ask the LLM to fix its mistake.
            print(f"[{self.name}] Code execution failed on attempt {attempt + 1}. Self-healing...")
            print(f"Error was: {output[:100]}...") # Print snippet of the error
            current_prompt = self._build_retry_prompt(code, output, attempt)
            
        # If we reach this point, the LLM failed to write working code 3 times
        raise RuntimeError(
            f"Agent {self.name} failed to generate working code after {self.max_retries} attempts."
        )

    def _extract_code(self, text) -> str:
        """
        Extract Python code from markdown code blocks in the LLM's response.
        Matches ```python ... ``` or just ``` ... ```
        """
        # --- Fix for newest Gemini models ---
        # Sometimes the LLM returns a list of content blocks instead of a plain string.
        # If it's a list, we extract the "text" from each block and join them into a string.
        if isinstance(text, list):
            string_parts = []
            for item in text:
                if isinstance(item, dict) and "text" in item:
                    string_parts.append(item["text"])
                else:
                    string_parts.append(str(item))
            text = "\n".join(string_parts)
            
        # Look for code between ```python and ```
        match = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
            
        # Fallback: look for general code blocks anywhere
        match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
            
        # If no code block is found, return empty string
        return ""

    def _build_retry_prompt(self, failed_code: str, error: str, attempt: int) -> str:
        """
        Create a prompt telling the LLM its code failed and asking it to fix it.
        """
        return RETRY_CODE_PROMPT.format(
            attempt=attempt + 1,
            failed_code=failed_code,
            error=error
        )
