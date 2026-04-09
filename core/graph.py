"""
Orchestrator — The manager that directs the agents using LangGraph.

WHAT THIS FILE DOES:
    Defines the workflow (the "Graph"). 
    It connects agents together and controls the flow of execution.

HOW IT WORKS:
    1. Define Nodes: Each agent becomes a "node" in the graph.
    2. Define Edges: We draw paths between the nodes.
    3. Compile: LangGraph turns this into a runnable application.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Import our shared state schema
from core.state import PipelineState

# Import our agents
from agents.data_profiler import DataProfilerAgent
from agents.data_cleaner import DataCleanerAgent


def build_graph():
    """Builds and returns the LangGraph workflow application."""
    
    # 1. Initialize the graph using our PipelineState notebook
    workflow = StateGraph(PipelineState)
    
    # 2. Instantiate our agents
    profiler_agent = DataProfilerAgent()
    cleaner_agent = DataCleanerAgent()
    
    # 3. Create wrapper functions for the nodes
    def run_profiler(state: PipelineState):
        print("\n" + "="*40)
        print("▶ PHASE: PROFILING")
        print("="*40)
        updates = profiler_agent.run(state)
        updates["current_phase"] = "profiling"
        return updates

    def run_cleaner_plan(state: PipelineState):
        print("\n" + "="*40)
        print("▶ PHASE: CLEANER (PLAN GENERATION)")
        print("="*40)
        updates = cleaner_agent.generate_plan(state)
        updates["current_phase"] = "cleaner_planning"
        return updates

    def run_cleaner_execute(state: PipelineState):
        print("\n" + "="*40)
        print("▶ PHASE: CLEANER (CODE EXECUTION)")
        print("="*40)
        updates = cleaner_agent.execute_plan(state)
        updates["current_phase"] = "cleaner_executing"
        return updates

    # 4. Add the nodes to the graph
    workflow.add_node("profiler", run_profiler)
    workflow.add_node("cleaner_plan", run_cleaner_plan)
    workflow.add_node("cleaner_execute", run_cleaner_execute)
    
    # 5. Define the edges (the flow path)
    workflow.set_entry_point("profiler")
    workflow.add_edge("profiler", "cleaner_plan")
    
    # Human-in-the-loop: 
    # Notice we simply draw an edge from 'cleaner_plan' to 'cleaner_execute'.
    # We will tell LangGraph to *pause* right before traversing it during `compile()`.
    workflow.add_edge("cleaner_plan", "cleaner_execute")
    workflow.add_edge("cleaner_execute", END)
    
    # 6. Compile the graph with Checkpointing and Interrupts
    # We need a memory saver to "freeze" the state while the human looks at it.
    memory = MemorySaver()
    app = workflow.compile(
        checkpointer=memory,
        interrupt_before=["cleaner_execute"]  # This tells LangGraph to pause exactly before this node!
    )
    
    return app
