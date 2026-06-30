# Auto-Analytics Agent

An autonomous, multi-agent data-science pipeline. Point it at a CSV and a team of
LLM-powered agents will **profile → clean → feature-engineer → select features →
train models → evaluate** the data end-to-end — pausing at key steps so a human can
review and adjust each plan before it runs.

Built with **LangGraph** (orchestration), **FastAPI** (backend), **scikit-learn**
(modeling), and **Next.js** (frontend).

---

## How It Works

The pipeline is a 7-node LangGraph graph sharing one typed state object. The two ⚡
markers are human-in-the-loop pause points where you approve the AI's plan:

```
profiler → cleaner_plan → ⚡ cleaner_execute → engineer_plan →
⚡ engineer_execute → engineer_select → model_trainer → evaluator → END
```

| Agent | Does |
|-------|------|
| **DataProfiler** | Profiles the dataset, resolves the target column and task type |
| **DataCleaner** | Plans + executes cleaning (validates that no missing values remain) |
| **FeatureEngineer** | Plans + executes encoding/scaling/feature creation, then selects features |
| **ModelTrainer** | Trains ≥3 models with cross-validation and saves them |
| **Evaluator** | Picks the best model, computes metrics, renders plots + an HTML report |

Each agent generates Python code with an LLM and runs it in an isolated subprocess
sandbox with a self-healing retry loop (it feeds its own tracebacks back to the LLM
to fix). If any agent reports an error, the graph short-circuits instead of cascading.

---

## Prerequisites

- **Python 3.11+** (developed on 3.13)
- **Node.js 18+** (for the frontend)
- An LLM API key (OpenAI or Google Gemini), or a local **Ollama** install

---

## Setup

### 1. Backend

```bash
# From the project root
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

### 2. Environment variables

```bash
copy .env.example .env         # Windows  (cp on macOS/Linux)
```

Then edit `.env`. The key settings:

| Variable | Purpose |
|----------|---------|
| `LLM_PROVIDER` | `openai`, `gemini`, or `ollama` |
| `OPENAI_API_KEY` / `GOOGLE_API_KEY` | API key for the chosen provider |
| `OPENAI_MODEL` / `GEMINI_MODEL` / `OLLAMA_MODEL` | Model name for the provider |
| `CODE_EXECUTION_TIMEOUT` | Max seconds for sandboxed code (default 120) |
| `MAX_RETRIES` | Self-healing retries per agent (default 3) |

> Using Ollama? Start it locally (`ollama serve`) and pull the model
> (`ollama pull llama3`) — no API key needed.

### 3. Frontend

```bash
cd frontend
npm install
```

---

## Running

Open two terminals from the project root.

**Terminal 1 — backend (FastAPI on :8000):**

```bash
venv\Scripts\activate
uvicorn api.main:app --reload --port 8000
```

API docs are auto-generated at <http://localhost:8000/docs>.

**Terminal 2 — frontend (Next.js on :3000):**

```bash
cd frontend
npm run dev
```

Open <http://localhost:3000>, enter a dataset path that exists on the **backend**
(e.g. `data/dummy_dirty.csv`), and step through the pipeline. At each review stage you
can type feedback (e.g. *"also drop the Name column"*) before approving.

---

## Using the API directly

```bash
# 1. Start a run (returns thread_id, data_profile, cleaning_plan)
curl -X POST http://localhost:8000/api/pipeline/start \
  -H "Content-Type: application/json" \
  -d '{"dataset_path": "data/dummy_dirty.csv", "task_type": "auto"}'

# 2. Approve the cleaning plan (returns cleaning_report, feature_plan)
curl -X POST http://localhost:8000/api/pipeline/<thread_id>/approve-cleaning \
  -H "Content-Type: application/json" \
  -d '{"human_feedback": "Looks good, proceed."}'

# 3. Approve the feature plan (runs FE + training + evaluation to completion)
curl -X POST http://localhost:8000/api/pipeline/<thread_id>/approve-features \
  -H "Content-Type: application/json" \
  -d '{"human_feedback": "Use MinMax scaling for Fare."}'
```

Generated artifacts are written to `models/` (trained `.joblib` models, train/test
CSVs, plots) and `reports/` (the HTML evaluation report), and are served by the
backend at `/models/...` and `/reports/...`.

---

## Project Structure

```
config/      Typed settings loaded from .env
core/        state.py (shared state), llm.py (provider factory),
             prompts.py (LLM templates), graph.py (LangGraph wiring)
agents/      base_agent.py + one agent per pipeline phase
tools/       code_executor.py (sandbox), file_manager.py (safe file I/O)
api/         main.py (FastAPI app + endpoints)
frontend/    Next.js app (5-stage UI in src/app/page.tsx)
tests/       pytest suite (unit + integration)
data/        input datasets and generated intermediate CSVs
models/      trained models, plots, train/test splits
reports/     generated HTML evaluation report
```

---

## Testing

```bash
# Fast unit tests (mocked LLM, no network)
venv\Scripts\python -m pytest -m "not integration"

# Full suite, including integration tests (these call the real LLM via your .env)
venv\Scripts\python -m pytest
```

---

## Notes & Limitations

- `MemorySaver` keeps graph state in-memory per process; swap in a persistent
  checkpointer (e.g. PostgresSaver) for production durability.
- Feature scaling currently fits on the full dataset; the train/test split happens in
  the Trainer. For zero leakage, fit scalers on the training split only and persist
  them as artifacts (planned).
