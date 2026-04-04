"""
API main router — Connects the LangGraph Orchestrator to the outside world.

WHAT THIS FILE DOES:
    Initializes FastAPI and creates an Endpoint (/api/run).
    When the Frontend sends a request to this endpoint, it triggers our Graph
    and returns the resulting Agent state back to the browser.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

from core.graph import build_graph

# Create the FastAPI app
app = FastAPI(title="Auto-Analytics Agent")

# Allow the Next.js frontend (which will run on port 3000) to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the expected JSON payload format
class RunPipelineRequest(BaseModel):
    dataset_path: str
    task_type: str = "auto"

# Compile the LangGraph orchestrator once when the severe starts
workflow_app = build_graph()

# Our single API endpoint
@app.post("/api/run")
def run_pipeline(request: RunPipelineRequest):
    """
    Endpoint that triggers the LangGraph workflow.
    """
    # Safety check
    if not os.path.exists(request.dataset_path):
         raise HTTPException(status_code=404, detail="Dataset file not found.")

    # Prepare user input for the graph
    initial_state = {
        "dataset_path": request.dataset_path,
        "task_type": request.task_type
    }
    
    try:
        # ▶ RUN THE AGENT PIPELINE!
        final_state = workflow_app.invoke(initial_state)
        
        # Send only the relevant parts of the state back to the frontend
        return {
            "status": "success",
            "phase": final_state.get("current_phase"),
            "data_profile": final_state.get("data_profile"),
            "errors": final_state.get("errors")
        }
        
    except Exception as e:
        # If the graph violently crashes, return a 500 server error
        raise HTTPException(status_code=500, detail=str(e))
