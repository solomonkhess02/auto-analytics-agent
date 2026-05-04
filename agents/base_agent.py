"""
Abstract Base Class for all pipeline agents.
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
        self.llm = get_llm()
        self.max_retries = 3

    @abstractmethod
    def run(self, state: PipelineState) -> dict:
        """
        Execute agent logic.
        
        Args:
            state: The current pipeline state.
            
        Returns:
            A dictionary containing the state updates to apply.
        """
        pass

    def _generate_and_execute_code(self, prompt: str) -> tuple[str, str]:
        """
        Agentic self-healing execution loop.
        
        Args:
            prompt: The instruction to send to the LLM.
            
        Returns:
            A tuple of (executed_code, output_of_code).
            
        Raises:
            RuntimeError: If the code fails after max_retries.
        """
        current_prompt = prompt
        
        for attempt in range(self.max_retries):
            response = self.llm.invoke(current_prompt)
            code = self._extract_code(response.content)
            
            if not code:
                print(f"[{self.name}] No code found in response on attempt {attempt + 1}")
                current_prompt = NO_CODE_FOUND_PROMPT.format(prompt=prompt)
                continue
                
            success, output = execute_code(code)
            
            if success:
                return code, output
                
            print(f"[{self.name}] Code execution failed on attempt {attempt + 1}. Self-healing...")
            print(f"Error was: {output[:100]}...")
            current_prompt = self._build_retry_prompt(code, output, attempt)
            
        raise RuntimeError(
            f"Agent {self.name} failed to generate working code after {self.max_retries} attempts."
        )

    def _extract_code(self, text) -> str:
        """
        Extract Python code from markdown code blocks in the LLM's response.
        """
        if isinstance(text, list):
            string_parts = []
            for item in text:
                if isinstance(item, dict) and "text" in item:
                    string_parts.append(item["text"])
                else:
                    string_parts.append(str(item))
            text = "\n".join(string_parts)
            
        match = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
            
        match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
            
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
