"""
Code Executor — Safely runs LLM-generated Python code in a sandbox.

WHAT THIS FILE DOES:
    When an agent generates Python code (e.g., pandas code to clean data),
    this tool executes it in a separate subprocess and returns the output.

WHY A SANDBOX:
    LLM-generated code could be buggy or dangerous. By running it in a
    subprocess with a timeout:
    - If it crashes → our main app keeps running
    - If it hangs → the timeout kills it after N seconds
    - It runs in isolation → can't access our app's memory

HOW IT WORKS:
    1. Write the code string to a temporary .py file
    2. Run: python temp_file.py (as a subprocess)
    3. Capture stdout (what the code prints) + stderr (errors)
    4. Return (success, output) tuple
    5. Clean up the temp file

USAGE:
    from tools.code_executor import execute_code

    success, output = execute_code("print(2 + 2)")
    # success = True, output = "4\n"

    success, output = execute_code("1/0")
    # success = False, output = "ZeroDivisionError: division by zero"
"""

import subprocess
import tempfile
import os
import sys


def execute_code(code: str, timeout: int = 120) -> tuple[bool, str]:
    """
    Execute Python code in a sandboxed subprocess.

    Args:
        code:    Python code to execute (as a string).
        timeout: Maximum seconds to wait before killing the process.

    Returns:
        A tuple of (success: bool, output: str)
        - success: True if code ran without errors (exit code 0)
        - output: Everything the code printed (stdout + stderr combined)

    Example:
        >>> success, output = execute_code("import pandas as pd; print(pd.__version__)")
        >>> print(success)  # True
        >>> print(output)   # "2.2.0\n"
    """

    # --- Step 1: Create a temporary .py file with the code ---
    # We write to the tmp/ folder in our project (not system temp)
    tmp_dir = os.path.join(os.getcwd(), "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # tempfile creates a unique filename so multiple executions don't collide
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        dir=tmp_dir,
        delete=False,       # Don't delete yet — we need to run it first
        encoding="utf-8",
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        # --- Step 2: Run the code as a subprocess ---
        # subprocess.run starts a NEW Python process (isolated from ours)
        #
        # Why sys.executable? It uses the same Python that's running our app
        # (the one in venv/), ensuring the code has access to all our
        # installed packages (pandas, sklearn, etc.)
        result = subprocess.run(
            [sys.executable, tmp_path],     # Command: python /tmp/script_xyz.py
            capture_output=True,            # Capture stdout and stderr
            text=True,                      # Return strings, not bytes
            timeout=timeout,                # Kill if it takes too long
            cwd=os.getcwd(),                # Run from project root directory
        )

        # --- Step 3: Combine stdout + stderr and return ---
        output = result.stdout
        if result.stderr:
            output += result.stderr

        # returncode == 0 means the script ran without errors
        return result.returncode == 0, output.strip()

    except subprocess.TimeoutExpired:
        # The code took too long — this is a safety net
        return False, f"Code execution timed out after {timeout} seconds"

    except Exception as e:
        # Something unexpected went wrong with subprocess itself
        return False, f"Execution error: {str(e)}"

    finally:
        # --- Step 4: Always clean up the temp file ---
        # 'finally' runs whether the code succeeded, failed, or timed out
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
