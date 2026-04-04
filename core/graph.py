"""
Orchestrator — The manager that directs the agents using LangGraph.

WHAT THIS FILE DOES:
    Defines the workflow (the "Graph"). 
    It connects agents together and controls the flow of execution.

    Currently (Phase 1): 
        START -> Profiler -> END

    Eventually (Phase 5): 
        START -> Profiler -> Cleaner -> Engineer -> Trainer -> Evaluator -> END

HOW IT WORKS:
    1. Define Nodes: Each agent becomes a "node" in the graph.
    2. Define Edges: We draw paths between the nodes.
    3. Compile: LangGraph turns this into a runnable application.
"""

from langgraph.graph import StateGraph, END

# Import our shared state schema
from core.state import PipelineState
# Import our agent
from agents.data_profiler import DataProfilerAgent


def build_graph():
    """Builds and returns the LangGraph workflow application."""
    
    # 1. Initialize the graph using our PipelineState notebook
    workflow = StateGraph(PipelineState)
    
    # 2. Instantiate our agents
    profiler_agent = DataProfilerAgent()
    
    # 3. Create wrapper functions for the nodes
    # LangGraph expects nodes to be functions that take 'state' and return a dict
    def run_profiler(state: PipelineState):
        print("\n" + "="*40)
        print("▶ PHASE: PROFILING")
        print("="*40)
        
        # Run the actual agent
        updates = profiler_agent.run(state)
        
        # Add metadata to the update
        updates["current_phase"] = "profiling"
        return updates

    # 4. Add the nodes to the graph
    workflow.add_node("profiler", run_profiler)
    
    # 5. Define the edges (the flow path)
    # Start at the profiler
    workflow.set_entry_point("profiler")
    
    # After the profiler finishes, go to the END (for now)
    workflow.add_edge("profiler", END)
    
    # 6. Compile the graph into an executable application
    app = workflow.compile()
    
    return app
