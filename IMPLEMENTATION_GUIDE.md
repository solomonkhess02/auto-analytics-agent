# 🧠 Autonomous Data Scientist Agent — Complete Implementation Guide

> **Project Codename:** `auto-analytics-agent`  
> **Goal:** Build a multi-agent AI system where a user uploads a dataset and the system autonomously cleans data, engineers features, selects & trains models, evaluates performance, and generates a human-readable report with visualizations.

---

## Table of Contents

1. [Vision & Problem Statement](#1-vision--problem-statement)
2. [Architecture Overview](#2-architecture-overview)
3. [Technology Stack](#3-technology-stack)
4. [Project Structure](#4-project-structure)
5. [Core Abstractions](#5-core-abstractions)
6. [Phase 1 — Foundation & Data Profiler Agent](#6-phase-1--foundation--data-profiler-agent)
7. [Phase 2 — Data Cleaner Agent](#7-phase-2--data-cleaner-agent)
8. [Phase 3 — Feature Engineer Agent](#8-phase-3--feature-engineer-agent)
9. [Phase 4 — Model Selector & Trainer Agent](#9-phase-4--model-selector--trainer-agent)
10. [Phase 5 — Evaluator, Reporter & Dashboard](#10-phase-5--evaluator-reporter--dashboard)
11. [Agent Communication & Orchestration](#11-agent-communication--orchestration)
12. [Code Execution & Safety](#12-code-execution--safety)
13. [Error Handling & Self-Healing](#13-error-handling--self-healing)
14. [Testing Strategy](#14-testing-strategy)
15. [Deployment](#15-deployment)
16. [Future Enhancements](#16-future-enhancements)
17. [Risk Mitigation](#17-risk-mitigation)

---

## 1. Vision & Problem Statement

### The Problem
A typical data science workflow involves 8-10 manual steps: loading data, profiling, cleaning, feature engineering, model selection, training, tuning, evaluation, and reporting. Each step requires domain expertise and significant time. Most of this workflow is repetitive and follows well-known patterns.

### The Solution
An **agentic AI system** that automates the entire workflow. The user uploads a CSV/Excel file, optionally specifies a target column and task type, and the system handles everything else — producing metrics, visualizations, and a natural language report explaining what it did and why.

### What Makes This "Very Impressive"
- **Not AutoML** — this isn't just `auto-sklearn`. Each agent *reasons* about the data, explains its decisions in natural language, and adapts its strategy based on what it observes.
- **Transparent decision-making** — every agent logs *why* it made each choice (e.g., "I chose to impute `age` with median because the distribution is right-skewed with 12% missing values").
- **Self-healing** — if generated code fails, the agent reads the traceback, diagnoses the issue, and retries with a fix.
- **Human-in-the-loop** — optional approval checkpoints after each phase.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     STREAMLIT FRONTEND                          │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────────────┐ │
│  │  Upload   │  │  Live Status │  │  Results Dashboard        │ │
│  │  Dataset  │  │  & Logs      │  │  (Metrics, Plots, Report) │ │
│  └────┬─────┘  └──────▲───────┘  └───────────▲───────────────┘ │
│       │               │                      │                  │
└───────┼───────────────┼──────────────────────┼──────────────────┘
        │               │                      │
        ▼               │                      │
┌───────────────────────┼──────────────────────┼──────────────────┐
│                    FASTAPI BACKEND                               │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              LangGraph Orchestrator                          ││
│  │                                                              ││
│  │  ┌──────────┐    ┌──────────┐    ┌──────────────────┐       ││
│  │  │ Data     │───▶│ Data     │───▶│ Feature          │       ││
│  │  │ Profiler │    │ Cleaner  │    │ Engineer         │       ││
│  │  └──────────┘    └──────────┘    └────────┬─────────┘       ││
│  │                                           │                  ││
│  │                                           ▼                  ││
│  │                  ┌──────────┐    ┌──────────────────┐       ││
│  │                  │Evaluator │◀───│ Model Selector   │       ││
│  │                  │& Reporter│    │ & Trainer        │       ││
│  │                  └──────────┘    └──────────────────┘       ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐      │
│  │ Code Sandbox │  │ File Manager │  │  LLM Interface   │      │
│  │ (exec engine)│  │ (datasets,   │  │  (OpenAI/Gemini) │      │
│  │              │  │  artifacts)  │  │                   │      │
│  └──────────────┘  └──────────────┘  └──────────────────┘      │
└──────────────────────────────────────────────────────────────────┘
```

### Data Flow
1. User uploads dataset via Streamlit → FastAPI receives and stores it
2. FastAPI triggers the LangGraph pipeline
3. Each agent node executes sequentially (with optional human-in-the-loop pauses)
4. Agents generate Python code → execute in sandbox → observe results → decide next action
5. Final results (metrics, plots, report) sent back to Streamlit for display

---

## 3. Technology Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **Orchestration** | LangGraph | Stateful graph-based agent orchestration with conditional edges, checkpoints, and human-in-the-loop support |
| **LLM** | OpenAI GPT-4o (primary), Google Gemini (fallback), Ollama (local dev) | Best reasoning capabilities for code generation and data analysis decisions |
| **Frontend** | Streamlit | Fastest path to a polished data-centric UI with built-in charts, file upload, and session state |
| **Backend API** | FastAPI | Async, fast, auto-docs, WebSocket support for live status streaming |
| **Data Processing** | Pandas, NumPy | Industry standard for tabular data manipulation |
| **ML** | Scikit-learn, XGBoost, LightGBM | Covers 95% of tabular ML use cases |
| **Visualization** | Matplotlib, Seaborn, Plotly | Static plots for reports + interactive plots for dashboard |
| **Code Execution** | `subprocess` + allowlisting (dev), E2B (production) | Safe execution of LLM-generated code |
| **State Persistence** | SQLite (dev), PostgreSQL (prod) | LangGraph checkpointer for pipeline state |
| **Config** | Pydantic Settings + `.env` | Type-safe configuration management |

### Python Version
- **Python 3.11+** (required for LangGraph and modern typing features)

### Key Dependencies (requirements.txt)
```
langgraph>=0.2.0
langchain>=0.3.0
langchain-openai>=0.2.0
langchain-google-genai>=2.0.0
fastapi>=0.115.0
uvicorn>=0.30.0
streamlit>=1.40.0
pandas>=2.2.0
numpy>=1.26.0
scikit-learn>=1.5.0
xgboost>=2.1.0
lightgbm>=4.5.0
matplotlib>=3.9.0
seaborn>=0.13.0
plotly>=5.24.0
pydantic>=2.9.0
pydantic-settings>=2.5.0
python-multipart>=0.0.12
httpx>=0.27.0
python-dotenv>=1.0.0
```

---

## 4. Project Structure

```
auto-analytics-agent/
│
├── .env.example                  # Environment variable template
├── .env                          # Actual env vars (gitignored)
├── .gitignore
├── requirements.txt
├── README.md
├── IMPLEMENTATION_GUIDE.md       # This document
│
├── config/
│   ├── __init__.py
│   └── settings.py               # Pydantic Settings — all config in one place
│
├── core/
│   ├── __init__.py
│   ├── state.py                  # PipelineState TypedDict — shared state schema
│   ├── orchestrator.py           # LangGraph StateGraph definition & compilation
│   ├── llm.py                    # LLM client factory (OpenAI / Gemini / Ollama)
│   └── prompts.py                # All system/user prompt templates
│
├── agents/
│   ├── __init__.py
│   ├── base_agent.py             # Abstract base class for all agents
│   ├── data_profiler.py          # Phase 1: Data Profiler Agent
│   ├── data_cleaner.py           # Phase 2: Data Cleaner Agent
│   ├── feature_engineer.py       # Phase 3: Feature Engineer Agent
│   ├── model_selector.py         # Phase 4: Model Selector & Trainer Agent
│   └── evaluator.py              # Phase 5: Evaluator & Reporter Agent
│
├── tools/
│   ├── __init__.py
│   ├── code_executor.py          # Sandboxed Python code execution
│   ├── file_manager.py           # Read/write datasets, artifacts, plots
│   ├── plot_generator.py         # Matplotlib/Seaborn/Plotly plot utilities
│   └── report_writer.py          # Markdown/HTML report generation
│
├── api/
│   ├── __init__.py
│   ├── main.py                   # FastAPI app entry point
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── pipeline.py           # POST /upload, GET /status, GET /results
│   │   └── health.py             # GET /health
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py            # Pydantic request/response schemas
│   └── websocket.py              # WebSocket for live pipeline status
│
├── ui/
│   ├── app.py                    # Streamlit main app
│   ├── pages/
│   │   ├── 1_upload.py           # Dataset upload page
│   │   ├── 2_profile.py          # Data profile viewer
│   │   ├── 3_pipeline.py         # Pipeline progress & logs
│   │   └── 4_results.py          # Results dashboard
│   └── components/
│       ├── sidebar.py            # Sidebar with config options
│       ├── metrics_card.py       # Styled metric display cards
│       └── plot_viewer.py        # Interactive plot viewer
│
├── reports/                      # Generated output directory
│   └── .gitkeep
│
├── data/                         # Uploaded datasets directory
│   └── .gitkeep
│
└── tests/
    ├── __init__.py
    ├── conftest.py               # Shared fixtures
    ├── test_agents/
    │   ├── test_data_profiler.py
    │   ├── test_data_cleaner.py
    │   ├── test_feature_engineer.py
    │   ├── test_model_selector.py
    │   └── test_evaluator.py
    ├── test_tools/
    │   ├── test_code_executor.py
    │   └── test_file_manager.py
    └── test_api/
        └── test_pipeline.py
```

---

## 5. Core Abstractions

### 5.1 PipelineState — The Shared Memory

This is the single most important data structure. Every agent reads from and writes to this state. LangGraph passes it between nodes automatically.

```python
# core/state.py
from typing import TypedDict, Annotated, Optional
import operator

class DataProfile(TypedDict):
    shape: tuple[int, int]
    columns: list[str]
    dtypes: dict[str, str]
    missing_values: dict[str, int]
    missing_percentages: dict[str, float]
    descriptive_stats: dict
    unique_counts: dict[str, int]
    sample_rows: list[dict]
    correlations: dict
    target_column: Optional[str]
    task_type: Optional[str]  # "classification" | "regression" | "auto"
    profiler_summary: str

class CleaningReport(TypedDict):
    actions_taken: list[str]
    columns_dropped: list[str]
    columns_modified: dict[str, str]
    rows_before: int
    rows_after: int
    missing_before: int
    missing_after: int
    cleaning_code: str
    cleaner_summary: str

class FeatureReport(TypedDict):
    features_created: list[str]
    features_dropped: list[str]
    encoding_applied: dict[str, str]
    scaling_applied: dict[str, str]
    feature_importances: dict[str, float]
    final_feature_list: list[str]
    feature_code: str
    engineer_summary: str

class TrainingResult(TypedDict):
    model_name: str
    model_type: str
    hyperparameters: dict
    training_time_seconds: float
    cross_val_scores: list[float]
    cross_val_mean: float

class EvaluationReport(TypedDict):
    best_model_name: str
    metrics: dict[str, float]
    confusion_matrix: Optional[list[list[int]]]
    classification_report: Optional[str]
    feature_importances: dict[str, float]
    plot_paths: list[str]
    natural_language_report: str
    report_html_path: str

class PipelineState(TypedDict):
    # --- Input ---
    dataset_path: str
    target_column: Optional[str]
    task_type: str
    user_instructions: Optional[str]
    # --- Agent Outputs ---
    data_profile: Optional[DataProfile]
    cleaning_report: Optional[CleaningReport]
    feature_report: Optional[FeatureReport]
    training_results: Optional[list[TrainingResult]]
    evaluation_report: Optional[EvaluationReport]
    # --- Pipeline Metadata ---
    messages: Annotated[list[dict], operator.add]
    current_phase: str
    errors: Annotated[list[str], operator.add]
    warnings: Annotated[list[str], operator.add]
    # --- Intermediate Data Paths ---
    cleaned_dataset_path: Optional[str]
    engineered_dataset_path: Optional[str]
    model_artifacts_dir: Optional[str]
```

### 5.2 Base Agent

```python
# agents/base_agent.py
from abc import ABC, abstractmethod
from core.state import PipelineState
from core.llm import get_llm
from tools.code_executor import execute_code

class BaseAgent(ABC):
    def __init__(self, name: str):
        self.name = name
        self.llm = get_llm()
        self.max_retries = 3

    @abstractmethod
    def run(self, state: PipelineState) -> dict:
        """Execute agent logic. Returns partial state update dict."""
        pass

    def _generate_and_execute_code(self, prompt: str, state: PipelineState) -> tuple[str, str]:
        """Ask LLM to generate code, execute in sandbox, return (code, output).
        Includes self-healing retry loop on failure."""
        for attempt in range(self.max_retries):
            response = self.llm.invoke(prompt)
            code = self._extract_code(response.content)
            success, output = execute_code(code, timeout=120)
            if success:
                return code, output
            prompt = self._build_retry_prompt(code, output, attempt)
        raise RuntimeError(f"Agent {self.name} failed after {self.max_retries} retries")

    def _extract_code(self, text: str) -> str:
        """Extract Python code from markdown code blocks."""
        pass  # parse ```python ... ``` blocks

    def _build_retry_prompt(self, failed_code: str, error: str, attempt: int) -> str:
        return f"""The following code failed. Fix it.
## Failed Code (Attempt {attempt + 1})
```python
{failed_code}
```
## Error
```
{error}
```
Generate corrected Python code only."""
```

### 5.3 LLM Client Factory

```python
# core/llm.py
from config.settings import settings
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

def get_llm(model_override: str = None):
    provider = settings.llm_provider
    model = model_override or settings.llm_model
    if provider == "openai":
        return ChatOpenAI(model=model, temperature=0.1, api_key=settings.openai_api_key)
    elif provider == "gemini":
        return ChatGoogleGenerativeAI(model=model, temperature=0.1, google_api_key=settings.google_api_key)
    elif provider == "ollama":
        return ChatOpenAI(model=model, temperature=0.1, base_url="http://localhost:11434/v1", api_key="ollama")
```

---

## 6. Phase 1 — Foundation & Data Profiler Agent

**Goal:** Set up the project skeleton and build the first agent that analyzes an uploaded dataset.

### 6.1 What to Build

| # | Task | Details |
|---|------|---------|
| 1 | Project scaffold | Create all directories, `requirements.txt`, `.env.example`, `config/settings.py` |
| 2 | Core state & LLM | Implement `PipelineState`, LLM factory, base agent |
| 3 | Code executor tool | Sandboxed Python code execution with timeout |
| 4 | File manager tool | Save/load datasets, manage artifact paths |
| 5 | Data Profiler Agent | The first agent node in the graph |
| 6 | Minimal FastAPI | `POST /upload` endpoint that triggers profiling |
| 7 | Minimal Streamlit | Upload page + profile viewer |

### 6.2 Data Profiler Agent — Detailed Design

**Input:** Raw dataset path from `PipelineState.dataset_path`

**Process:**
1. Load the dataset with pandas
2. Compute statistical profile (shape, dtypes, missing values, correlations, unique counts, descriptive stats)
3. Send profile to LLM with a prompt asking for:
   - Natural language summary of the dataset
   - Identification of potential target columns
   - Detection of data quality issues
   - Recommendation of task type (classification vs regression)
4. If `target_column` not provided by user, use LLM's recommendation

**Output:** Updates `state.data_profile` with a complete `DataProfile` dict

**Key Prompt Template:**
```
You are a senior data scientist analyzing a new dataset.

## Dataset Overview
- Shape: {shape}
- Columns: {columns_with_dtypes}
- Missing Values: {missing_summary}
- Sample Rows: {sample_rows}
- Descriptive Statistics: {desc_stats}

## Your Task
1. Write a clear, concise summary of this dataset
2. If no target column was specified, recommend one and explain why
3. Determine if this is a classification or regression problem
4. List the top 3 data quality concerns
5. Rate the overall data quality on a scale of 1-10

Respond in JSON format with keys: summary, recommended_target, task_type, quality_concerns, quality_score
```

### 6.3 Minimal Orchestrator (Phase 1)

```python
from langgraph.graph import StateGraph, START, END
from core.state import PipelineState

def build_pipeline():
    profiler = DataProfilerAgent()
    graph = StateGraph(PipelineState)
    graph.add_node("profiler", profiler.run)
    graph.add_edge(START, "profiler")
    graph.add_edge("profiler", END)
    return graph.compile()
```

### 6.4 Phase 1 Deliverables
- ✅ Upload a CSV → see a complete data profile
- ✅ LLM-generated natural language summary
- ✅ Auto-detected target column and task type
- ✅ Data quality score and concerns listed
- ✅ Streamlit page showing profile info with basic charts

---

## 7. Phase 2 — Data Cleaner Agent

**Goal:** Autonomous data cleaning with LLM-decided strategy.

### 7.1 Data Cleaner Agent — Detailed Design

**Input:** `state.data_profile` + raw dataset

**Process:**
1. LLM receives the data profile and generates a **cleaning plan** in structured JSON:
   ```json
   {
     "drop_columns": ["id", "unnamed_0"],
     "drop_columns_reason": "ID columns provide no predictive value",
     "handle_missing": {
       "age": {"strategy": "median", "reason": "Right-skewed, 5% missing"},
       "city": {"strategy": "mode", "reason": "Categorical, 2% missing"},
       "income": {"strategy": "drop_rows", "reason": "Only 0.3% missing"}
     },
     "handle_duplicates": {"action": "drop", "reason": "23 exact duplicates found"},
     "type_corrections": {
       "date_joined": {"from": "object", "to": "datetime64"},
       "zipcode": {"from": "int64", "to": "object"}
     },
     "outlier_handling": {
       "salary": {"strategy": "clip_iqr", "reason": "Extreme outliers beyond 3x IQR"}
     }
   }
   ```
2. LLM generates executable Python (pandas) code implementing the plan
3. Code executes in sandbox
4. If execution fails → self-healing retry loop
5. Post-cleaning validation: re-profile cleaned data, confirm improvements
6. Save cleaned dataset to `data/cleaned_{timestamp}.csv`

**Output:** Updates `state.cleaning_report` + `state.cleaned_dataset_path`

### 7.2 Human-in-the-Loop Checkpoint

After generating the cleaning plan but BEFORE executing:
- LangGraph pauses at a checkpoint
- Streamlit shows the plan to the user
- User can: ✅ Approve, ✏️ Modify, or ❌ Skip

### 7.3 Phase 2 Deliverables
- ✅ LLM generates a structured cleaning plan based on data profile
- ✅ User can review/modify the plan before execution
- ✅ Agent generates and executes real pandas code
- ✅ Self-healing on code errors
- ✅ Before/after comparison shown in Streamlit

---

## 8. Phase 3 — Feature Engineer Agent

**Goal:** Automatic feature creation, encoding, scaling, and selection.

### 8.1 Feature Engineer Agent — Detailed Design

**Input:** Cleaned dataset + data profile + task type

**Process:**
1. LLM analyzes columns and proposes feature engineering strategy:
   - **Datetime features:** Extract year, month, day, day_of_week, is_weekend, quarter
   - **Text features:** Length, word count, TF-IDF (if text columns detected)
   - **Numerical interactions:** Ratios, products for highly correlated pairs
   - **Categorical encoding:** One-hot (low cardinality ≤10), target encoding (high cardinality), ordinal (ordered)
   - **Numerical scaling:** StandardScaler, MinMaxScaler, or RobustScaler based on distribution
   - **Binning:** Discretize continuous variables where appropriate
2. LLM generates executable code for all transformations
3. Execute in sandbox
4. **Feature Selection** (post-engineering):
   - Compute feature importances using a quick Random Forest
   - Drop features with near-zero importance
   - Check for multicollinearity (VIF > 10 → drop)
   - LLM reviews and finalizes feature list

**Output:** Updates `state.feature_report` + `state.engineered_dataset_path`

### 8.2 Key Prompt for Feature Engineering
```
You are a senior ML engineer performing feature engineering.

## Dataset Info
- Task: {task_type} (target: {target_column})
- Columns after cleaning: {columns_with_dtypes}
- Sample data: {sample_rows}
- Correlations with target: {target_correlations}

## Rules
1. Generate Python code using pandas and sklearn preprocessing
2. Read from '{cleaned_path}' and save to '{engineered_path}'
3. Do NOT touch the target column
4. Store fitted encoders/scalers as pickle files for later inference
5. Print a summary of all transformations applied
```

### 8.3 Phase 3 Deliverables
- ✅ Automatic feature creation based on column types
- ✅ Proper encoding of all categoricals
- ✅ Numerical scaling with appropriate scaler selection
- ✅ Preliminary feature importance ranking
- ✅ Multicollinearity check
- ✅ Streamlit page showing feature engineering summary

---

## 9. Phase 4 — Model Selector & Trainer Agent

**Goal:** Automatically select appropriate models, train them, and find the best one.

### 9.1 Model Selection Strategy

| Signal | Decision |
|--------|----------|
| Classification, balanced | LogReg, RandomForest, XGBoost, LightGBM |
| Classification, imbalanced | Same + `class_weight='balanced'` or SMOTE |
| Regression, linear | LinearReg, Ridge, Lasso, ElasticNet |
| Regression, non-linear | RandomForest, XGBoost, LightGBM, SVR |
| Small dataset (< 1K rows) | Simpler models preferred |
| Large dataset (> 100K rows) | LightGBM preferred for speed |
| High dimensionality | Lasso/ElasticNet for built-in feature selection |

### 9.2 Training Process

```
For each candidate model:
  1. LLM defines a hyperparameter search space
  2. 5-fold cross-validation with defaults
  3. If promising, run RandomizedSearchCV (20 iterations)
  4. Record: model name, best params, CV scores, training time
  5. Save trained model as pickle

Best model selected by:
  - Classification → F1-score (macro for multiclass, binary for binary)
  - Regression → R² score (RMSE as secondary)
```

### 9.3 Phase 4 Deliverables
- ✅ LLM-selected candidate models with reasoning
- ✅ Cross-validation for all candidates
- ✅ Hyperparameter tuning for top candidates
- ✅ Best model selection with explanation
- ✅ All models saved as pickle files
- ✅ Training results table in Streamlit

---

## 10. Phase 5 — Evaluator, Reporter & Dashboard

**Goal:** Comprehensive evaluation, visualization, and natural language report.

### 10.1 Evaluator Agent Outputs

**For Classification:**
- Confusion matrix heatmap
- Classification report (precision, recall, F1 per class)
- ROC curve with AUC
- Precision-Recall curve
- Feature importance bar chart (top 20)
- Learning curve

**For Regression:**
- Actual vs Predicted scatter plot
- Residual plot + distribution
- Feature importance bar chart (top 20)
- Prediction error plot

### 10.2 Report Generator

The LLM generates a comprehensive markdown report covering:
1. Dataset Overview (from profiler)
2. Data Cleaning Summary (from cleaner)
3. Feature Engineering Summary (from engineer)
4. Model Selection & Training Results (from trainer)
5. Evaluation Metrics & Plots (from evaluator)
6. Key Insights (LLM-generated)
7. Recommendations (LLM-generated)

### 10.3 Streamlit Dashboard Pages

1. **Upload Page** — File upload, target column selector, task type, instructions textarea
2. **Pipeline Progress** — Real-time status via WebSocket, log viewer, phase indicators
3. **Data Profile** — Interactive profile viewer with charts, correlation heatmap
4. **Results Dashboard** — Metrics cards, interactive plots, model comparison table, downloadable report, code viewer

### 10.4 Phase 5 Deliverables
- ✅ Complete evaluation metrics and visualizations
- ✅ LLM-generated natural language report
- ✅ Downloadable HTML report with embedded plots
- ✅ Full Streamlit dashboard with all pages
- ✅ Real-time pipeline progress via WebSocket
- ✅ End-to-end demo working with sample datasets

---

## 11. Agent Communication & Orchestration

### 11.1 Full LangGraph Pipeline

```python
# core/orchestrator.py (Final version)
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

def build_pipeline():
    profiler = DataProfilerAgent()
    cleaner = DataCleanerAgent()
    engineer = FeatureEngineerAgent()
    trainer = ModelSelectorAgent()
    evaluator = EvaluatorAgent()

    graph = StateGraph(PipelineState)

    graph.add_node("profiler", profiler.run)
    graph.add_node("cleaner", cleaner.run)
    graph.add_node("engineer", engineer.run)
    graph.add_node("trainer", trainer.run)
    graph.add_node("evaluator", evaluator.run)

    graph.add_edge(START, "profiler")
    graph.add_conditional_edges(
        "profiler",
        route_after_profiling,
        {"continue": "cleaner", "abort": END}
    )
    graph.add_edge("cleaner", "engineer")
    graph.add_edge("engineer", "trainer")
    graph.add_edge("trainer", "evaluator")
    graph.add_edge("evaluator", END)

    checkpointer = SqliteSaver.from_conn_string("pipeline_state.db")
    return graph.compile(checkpointer=checkpointer)

def route_after_profiling(state: PipelineState) -> str:
    profile = state.get("data_profile")
    if profile and profile.get("quality_score", 0) > 2:
        return "continue"
    return "abort"
```

### 11.2 State Flow

```
START → [Profiler] writes data_profile
  │ (conditional: quality_score > 2?)
  ▼
[Cleaner] reads data_profile → writes cleaning_report, cleaned_dataset_path
  ▼
[Engineer] reads cleaned_dataset_path → writes feature_report, engineered_dataset_path
  ▼
[Trainer] reads engineered_dataset_path → writes training_results, model_artifacts_dir
  ▼
[Evaluator] reads ALL state → writes evaluation_report
  ▼
END
```

---

## 12. Code Execution & Safety

### 12.1 Development Mode — Subprocess Sandbox

```python
# tools/code_executor.py
import subprocess, tempfile, os

def execute_code(code: str, timeout: int = 120) -> tuple[bool, str]:
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir='./tmp') as f:
        f.write(code)
        tmp_path = f.name
    try:
        result = subprocess.run(
            ['python', tmp_path],
            capture_output=True, text=True, timeout=timeout,
            cwd=os.getcwd()
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, f"Timed out after {timeout}s"
    finally:
        os.unlink(tmp_path)
```

### 12.2 Production Mode — E2B Sandbox
For production, replace subprocess with E2B Firecracker microVMs for hardware-level isolation.

### 12.3 Safety Rules
- **Allowlisted imports only:** pandas, numpy, sklearn, xgboost, lightgbm, matplotlib, seaborn, plotly, json, pickle, time
- **File access restricted** to `./data/` and `./reports/`
- **Timeout:** 120 seconds max per execution
- **Memory limit:** 2GB per execution

---

## 13. Error Handling & Self-Healing

### 13.1 Self-Healing Loop
```
1. LLM generates code → execute in sandbox
   ├─ SUCCESS → parse output, update state
   └─ FAILURE → read traceback
2. Build retry prompt with: original task + failed code + traceback
3. LLM generates fixed code → execute again
4. After 3 failures → apply safe fallback, log error, continue pipeline
```

### 13.2 Fallback Strategies Per Agent

| Agent | Fallback |
|-------|----------|
| Profiler | Use raw `df.describe()` + `df.info()` without LLM summary |
| Cleaner | Drop null rows + drop duplicates (safe defaults) |
| Engineer | One-hot encode categoricals, StandardScaler on numerics |
| Trainer | Train only LogisticRegression / LinearRegression with defaults |
| Evaluator | Compute basic accuracy/RMSE only, skip plots |

---

## 14. Testing Strategy

### 14.1 Test Datasets (in `tests/fixtures/`)
1. **`iris_dirty.csv`** — Iris with injected nulls, duplicates, wrong types
2. **`housing_messy.csv`** — Regression with mixed types, text columns
3. **`churn_imbalanced.csv`** — Binary classification with 90/10 imbalance

### 14.2 Test Levels

| Level | What | How |
|-------|------|-----|
| **Unit** | Agent logic without LLM | Mock LLM responses, test parsing & state updates |
| **Integration** | Agent + code executor | Real LLM, small datasets |
| **E2E** | Full pipeline | Upload CSV via API, verify report |
| **Prompt** | LLM output quality | Assert JSON structure, valid Python |

### 14.3 Commands
```bash
pytest tests/ -m "not integration" -v    # Unit (fast)
pytest tests/ -m "integration" -v        # Integration (needs API key)
pytest tests/test_api/test_pipeline.py   # E2E
```

---

## 15. Deployment

### 15.1 Local Development
```bash
# Terminal 1: Backend
uvicorn api.main:app --reload --port 8000

# Terminal 2: Frontend
streamlit run ui/app.py --server.port 8501
```

### 15.2 Docker Compose
```yaml
version: '3.8'
services:
  backend:
    build: { context: ., dockerfile: Dockerfile.backend }
    ports: ["8000:8000"]
    environment: [OPENAI_API_KEY=${OPENAI_API_KEY}]
    volumes: [./data:/app/data, ./reports:/app/reports]
  frontend:
    build: { context: ., dockerfile: Dockerfile.frontend }
    ports: ["8501:8501"]
    environment: [BACKEND_URL=http://backend:8000]
    depends_on: [backend]
```

### 15.3 Cloud Options
- **Railway / Render** — Easiest for hobby projects
- **AWS ECS + Fargate** — Production scale
- **GCP Cloud Run** — Serverless, pay-per-use

---

## 16. Future Enhancements

| Priority | Enhancement | Description |
|----------|------------|-------------|
| 🔥 High | Deep Learning Support | PyTorch/TF agents for image/text/time-series |
| 🔥 High | Multi-dataset Support | Join/merge multiple datasets |
| 🟡 Medium | AutoML Comparison | Compare agent vs auto-sklearn/TPOT baselines |
| 🟡 Medium | Model Deployment | One-click deploy model as REST API |
| 🟡 Medium | RAG Domain Knowledge | Upload docs → better feature engineering |
| 🟢 Low | Conversation Mode | Chat with agent about results |
| 🟢 Low | Time-Series Support | Specialized forecasting agent |
| 🟢 Low | PDF Report Export | Polished PDF via WeasyPrint |

---

## 17. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| LLM generates dangerous code | High | Sandbox, import allowlist, file restrictions |
| Incorrect analysis | Medium | Validation checks, human-in-the-loop approvals |
| API key costs | Medium | Token tracking, model fallback (GPT-4o → mini → local) |
| Pipeline hangs on large data | Medium | Timeouts, size limits, sampling for profiling |
| LLM hallucinates metrics | Medium | All metrics from actual sklearn, never LLM text |
| Context window overflow | Low | Summarize dataframes, send only samples + stats |

---

## Quick Reference: Getting Started

```bash
# 1. Clone and setup
cd "d:\IIT CODES\PROJECTS\AG auto-analytics-agent"
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure
copy .env.example .env
# Edit .env with your API keys

# 3. Run
uvicorn api.main:app --reload --port 8000
streamlit run ui/app.py --server.port 8501

# 4. Test
pytest tests/ -v
```

---

> **This document is the single source of truth for the project.** Reference it in any conversation for full context on architecture, design decisions, and implementation details.
