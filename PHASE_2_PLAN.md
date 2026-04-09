# 🧠 Phase 2: The "Agentic" Data Cleaner — Learning Roadmap

> **Goal:** Evolve our system from a simple "automation script" into a true **AI Agent** that can plan, pause for human feedback, write code dynamically, and validate its own work.

In Phase 1, we learned the basics of LangGraph (State, Agents, Tools). In Phase 2, we will dive deep into **Agentic Design Patterns**. We will build this in 5 distinct steps.

---

### Step 8: Centralizing the Brain (Prompt Management) 🧠
**Concept:** In Phase 1, we hardcoded our LLM instructions inside `data_profiler.py`. As agents get more complex (learning to self-heal, handle feedback, and validate), their prompts become massive. A best practice in AI engineering is separating "prompts" from "Python execution logic".

**What we'll build:**
- Create `core/prompts.py`
- Move Phase 1 prompts (`PROFILER_CODE_PROMPT`, etc.) into this file.
- Write new, highly detailed prompts for the Data Cleaner Agent.

**What you'll learn:**
- How to structure complex prompts using variables (e.g. `{dataset_shape}`, `{missing_values}`).
- The difference between "System Prompts" (who you are) and "Human Prompts" (what to do now).

---

### Step 9: Agentic Reasoning (Plan Generation) 📝
**Concept:** A script just "does things". An Agent **plans** things first. Before writing a single line of pandas code, our Cleaner Agent must look at the Data Profile, figure out what's wrong, and draft a plan.

**What we'll build:**
- Create `agents/data_cleaner.py` and implement the `generate_plan` method.
- The LLM will consume the `data_profile` and output a strictly formatted JSON document (e.g., `{"drop_columns": ["id"], "impute": {"age": "median"}}`).

**What you'll learn:**
- Prompting LLMs for guaranteed JSON output (even without OpenAI's structured outputs).
- How forcing an AI to "think step-by-step" and build a plan drastically improves its coding accuracy later.

---

### Step 10: Human-in-the-Loop Orchestration 🚦
**Concept:** The best agents are "Co-pilots". Now that we have a plan, we shouldn't execute it blindly. We must pause the LangGraph pipeline, show the plan to the human, and wait for approval or modification (e.g., "Actually, drop the `age` column instead of imputing it").

**What we'll build:**
- Update `core/state.py` to add `cleaning_plan` and `human_feedback` to our `PipelineState`.
- Update `core/graph.py` to split the Cleaner node into two: `plan_cleaner` and `execute_cleaner`.
- Use LangGraph's `interrupt_before` to pause execution right before `execute_cleaner`.

**What you'll learn:**
- How LangGraph captures and freezes state "in time" natively.
- How to explicitly resume a halted LangGraph pipeline with injected human feedback.

---

### Step 11: Dynamic Execution & The Validation Loop 🔁
**Concept:** Now the Agent has an approved (or human-modified) plan. It will generate pandas code and run it. But a true agent doesn't stop there. What if the code ran successfully, but the missing values are *still* there? The agent must *verify* its work.

**What we'll build:**
- Add `execute_plan` methodology to `agents/data_cleaner.py`.
- Feed the JSON plan + Human Feedback to the LLM to generate Python code.
- Execute it using our sandbox tool.
- **The Validation Loop:** Write logic that checks `cleaned_df.isna().sum()`. If there are still nulls, bounce it back to the LLM automatically saying "You failed to clean all nulls, fix your code!"

**What you'll learn:**
- Self-correction patterns: "Code execution success" vs "Business logic success".
- Passing human feedback into an LLM context window effectively.

---

### Step 12: Integration Testing for Agents 🧪
**Concept:** Testing AI systems is notoriously hard because LLM outputs are non-deterministic. How do we test that the Cleaner Agent works? We give it a deliberately terrible dataset and trap it in the sandbox.

**What we'll build:**
- Create `tests/test_agents/test_data_cleaner.py`.
- Create a dummy "dirty" dataset (mixed types, lots of nulls).
- Write a test script mimicking the LangGraph flow (with a mock human feedback interaction).

**What you'll learn:**
- How to mock LLM outputs for faster unit testing.
- How to structure end-to-end integration tests for multi-agent workflows.

---

### Ready to start?
To begin with **Step 8: Centralizing the Brain**, we will create `core/prompts.py` and move the existing prompts out of the base agent and profiler agent. Let me know when you're ready!
