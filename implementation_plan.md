# 🧠 Autonomous Data Scientist Agent — Learning-First Implementation Plan

> **Goal:** Build a multi-agent AI system step-by-step, understanding every concept along the way.  
> **Frontend:** Next.js (deployed on Vercel)  
> **Backend:** FastAPI + LangGraph (Python)  
> **Approach:** One concept at a time. Learn → Build → Test → Move on.

---

## Architecture Change: Streamlit → Next.js

> [!IMPORTANT]
> The original IMPLEMENTATION_GUIDE.md specified Streamlit. We're replacing it with **Next.js** for a more professional frontend. This changes the project structure:

```
BEFORE (Streamlit):                    AFTER (Next.js):
─────────────────                      ────────────────
ui/                                    frontend/          ← Next.js app
  app.py                                src/
  pages/                                  app/            ← Next.js App Router
    1_upload.py                             page.tsx      ← Home/Upload page
    2_profile.py                            pipeline/
  components/                                 page.tsx    ← Pipeline progress
    sidebar.py                              results/
    metrics_card.py                           page.tsx    ← Results dashboard
                                          components/     ← React components
                                          lib/            ← API client, utils
                                        package.json
                                        tailwind.config.ts
```

**What stays the same:** The entire Python backend (FastAPI, LangGraph, agents, tools) is unchanged. The frontend just calls the API differently — Next.js uses `fetch()` instead of Streamlit's built-in widgets.

---

## Learning Roadmap

We'll build this in **7 steps**, each focused on one core concept. No rushing.

---

### Step 1: State — The Shared Brain 🧠
**Concept:** In an agentic pipeline, multiple agents need to pass information to each other. They do this through a **shared state object** — think of it as a notebook that each agent reads from and writes to.

**What we'll build:**
- `core/state.py` — The `PipelineState` TypedDict

**What you'll learn:**
- Why agents can't just use regular variables
- What a TypedDict is and why it's used here
- How LangGraph automatically passes state between agents
- What `Annotated[list, operator.add]` means (append-only state fields)

**Files:** 1 file (~80 lines)

---

### Step 2: LLM Client — Talking to AI Models 🤖
**Concept:** Every agent needs to ask an LLM (like GPT-4o or Gemini) for help — to analyze data, generate code, or make decisions. We need a clean way to create these LLM connections.

**What we'll build:**
- `core/llm.py` — LLM client factory function

**What you'll learn:**
- What LangChain's `ChatOpenAI` / `ChatGoogleGenerativeAI` wrappers do
- The factory pattern — switching between providers with one config change
- What "temperature" means and why we use 0.1 (low creativity, high consistency)
- How to test that your API key works

**Files:** 1 file (~30 lines) + a quick test

---

### Step 3: Tools — What Agents Can *Do* 🔧
**Concept:** An LLM by itself can only generate text. To be useful, agents need **tools** — functions they can call to interact with the real world (execute code, read files, save results).

**What we'll build:**
- `tools/code_executor.py` — Sandboxed Python code execution
- `tools/file_manager.py` — Read/write datasets and artifacts

**What you'll learn:**
- Why LLM-generated code must run in a sandbox (safety!)
- How `subprocess` isolates code execution
- What a "tool" means in the LangChain/LangGraph world
- The difference between a tool and a regular function

**Files:** 2 files (~120 lines total)

---

### Step 4: Base Agent — The Agent Blueprint 🏗️
**Concept:** All our agents (Profiler, Cleaner, Trainer, etc.) follow the same pattern: receive state → ask LLM → execute code → update state. We define this pattern once in a base class.

**What we'll build:**
- `agents/base_agent.py` — Abstract base class with self-healing loop

**What you'll learn:**
- What makes something an "agent" vs just an LLM call (the reasoning + action loop)
- The ReAct pattern: Reason → Act → Observe → Repeat
- Self-healing: what happens when LLM-generated code fails
- How retry/fallback logic works

**Files:** 1 file (~80 lines)

---

### Step 5: First Agent — Data Profiler 📊
**Concept:** Our first real agent! It takes a raw dataset, analyzes it (shape, types, missing values, correlations), and asks the LLM to summarize findings and recommend next steps.

**What we'll build:**
- `core/prompts.py` — Prompt templates
- `agents/data_profiler.py` — The Data Profiler Agent

**What you'll learn:**
- How to write effective prompts for data analysis
- How an agent reads state, does work, and writes back to state
- Structured LLM output (getting JSON back, not just freeform text)
- Your first end-to-end agent run (input CSV → structured profile output)

**Files:** 2 files (~200 lines total)

---

### Step 6: Orchestration — Connecting Agents into a Pipeline 🔗
**Concept:** Individual agents are useful, but the real power comes from **connecting them into a graph**. LangGraph lets us define: "run Profiler first, then if data quality is OK, run Cleaner, then Engineer, then Trainer, then Evaluator."

**What we'll build:**
- `core/orchestrator.py` — LangGraph StateGraph (starting with just Profiler)

**What you'll learn:**
- What a **state graph** is (nodes = agents, edges = connections)
- Conditional edges: branching based on agent output
- Checkpointing: saving pipeline state so you can resume later
- Human-in-the-loop: pausing the pipeline for user approval
- How to run the full pipeline from Python

**Files:** 1 file (~60 lines) + testing it end-to-end

---

### Step 7: API + Frontend — Making It Real 🌐
**Concept:** Now we expose the pipeline as a web API (FastAPI) and build a professional frontend (Next.js) so users can upload datasets and see results in a beautiful UI.

**What we'll build:**
- `api/main.py` — FastAPI app
- `api/routes/pipeline.py` — Upload + status + results endpoints
- `api/routes/health.py` — Health check
- `api/models/schemas.py` — Request/response schemas
- `api/websocket.py` — Live pipeline status streaming
- `frontend/` — Full Next.js app with upload, progress, and results pages

**What you'll learn:**
- How FastAPI serves as the bridge between frontend and pipeline
- WebSocket for real-time progress updates
- Next.js App Router, React components, and API calls
- How to wire everything together into a working product

**Files:** Many files — but we'll do it piece by piece

---

## Updated Dependencies

> [!WARNING]
> **Streamlit is being removed** from `requirements.txt`. The frontend is now a separate **Next.js** project in `frontend/` with its own `package.json`.

### Python (requirements.txt) changes:
```diff
- streamlit>=1.40.0
```

### New (frontend/package.json):
```
next, react, react-dom, typescript, tailwindcss,
axios (HTTP client), recharts (charts), lucide-react (icons)
```

---

## Removed Directories

The old Streamlit `ui/` directory structure will be replaced:

```diff
- ui/
-   __init__.py
-   pages/__init__.py
-   components/__init__.py
+ frontend/     ← Next.js project (created via create-next-app)
```

---

## Open Questions

> [!IMPORTANT]
> Please confirm these before we begin:

1. **Which LLM provider will you use?** You need an API key for at least one:
   - OpenAI (`OPENAI_API_KEY`) — recommended, best code generation
   - Google Gemini (`GOOGLE_API_KEY`) — good free tier
   - Ollama (local, free, but slower) — no API key needed

2. **Do you have Node.js installed?** We need it for the Next.js frontend. Run `node --version` in your terminal — you need v18+.

3. **Ready to start with Step 1 (State)?** Once you confirm, I'll walk you through the concept first, then we'll build `core/state.py` together.
