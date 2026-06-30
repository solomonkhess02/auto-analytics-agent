"""
FastAPI router for LangGraph orchestration.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import uuid

from core.graph import build_graph

# Ensure directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

app = FastAPI(title="Auto-Analytics Agent API", version="2.0.0")

app.mount("/models", StaticFiles(directory="models"), name="models")
app.mount("/reports", StaticFiles(directory="reports"), name="reports")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

workflow_app = build_graph()
active_sessions: dict[str, dict] = {}


class StartPipelineRequest(BaseModel):
    dataset_path: str
    task_type: str = "auto"

class ApproveStepRequest(BaseModel):
    human_feedback: str = "Looks good, proceed."


@app.post("/api/pipeline/start")
def start_pipeline(request: StartPipelineRequest):
    """
    Initializes a new pipeline run.
    """
    if not os.path.exists(request.dataset_path):
        raise HTTPException(status_code=404, detail=f"Dataset not found: '{request.dataset_path}'")

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    active_sessions[thread_id] = config

    initial_state = {
        "dataset_path": request.dataset_path,
        "task_type": request.task_type,
        "messages": [],
        "errors": [],
        "warnings": [],
    }

    try:
        for _ in workflow_app.stream(initial_state, config=config, stream_mode="values"):
            pass

        state = workflow_app.get_state(config)

        errors = state.values.get("errors", [])
        if errors:
            raise HTTPException(status_code=500, detail=f"Pipeline error: {errors}")

        return {
            "thread_id": thread_id,
            "paused_before": list(state.next),
            "data_profile": state.values.get("data_profile"),
            "cleaning_plan": state.values.get("cleaning_plan"),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/pipeline/{thread_id}/approve-cleaning")
def approve_cleaning(thread_id: str, request: ApproveStepRequest):
    """
    Resumes graph execution after cleaner plan approval.
    """
    if thread_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    config = active_sessions[thread_id]

    try:
        workflow_app.update_state(config, {"human_feedback": request.human_feedback}, as_node="cleaner_plan")

        for _ in workflow_app.stream(None, config=config, stream_mode="values"):
            pass

        state = workflow_app.get_state(config)

        errors = state.values.get("errors", [])
        if errors:
            raise HTTPException(status_code=500, detail=f"Pipeline error: {errors}")

        return {
            "thread_id": thread_id,
            "paused_before": list(state.next),
            "cleaning_report": state.values.get("cleaning_report"),
            "feature_plan": state.values.get("feature_plan"),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/pipeline/{thread_id}/approve-features")
def approve_features(thread_id: str, request: ApproveStepRequest):
    """
    Resumes graph execution after feature plan approval.
    """
    if thread_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    config = active_sessions[thread_id]

    try:
        workflow_app.update_state(config, {"human_feedback": request.human_feedback}, as_node="engineer_plan")

        for _ in workflow_app.stream(None, config=config, stream_mode="values"):
            pass

        state = workflow_app.get_state(config)

        errors = state.values.get("errors", [])
        if errors:
            raise HTTPException(status_code=500, detail=f"Pipeline error: {errors}")

        return {
            "thread_id": thread_id,
            "paused_before": [],
            "feature_report": state.values.get("feature_report"),
            "engineered_dataset_path": state.values.get("engineered_dataset_path"),
            "training_results": state.values.get("training_results"),
            "evaluation_report": state.values.get("evaluation_report"),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # This is the terminal step of the pipeline; drop the session regardless
        # of success or failure so abandoned/errored runs don't leak.
        active_sessions.pop(thread_id, None)


@app.get("/api/health")
def health():
    return {"status": "ok", "active_sessions": len(active_sessions)}
