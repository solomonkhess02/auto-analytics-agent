"""
LangGraph orchestrator workflow definition.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from core.state import PipelineState
from agents.data_profiler import DataProfilerAgent
from agents.data_cleaner import DataCleanerAgent
from agents.feature_engineer import FeatureEngineerAgent


def build_graph():
    """Builds and returns the LangGraph workflow application."""
    
    workflow = StateGraph(PipelineState)
    
    profiler_agent = DataProfilerAgent()
    cleaner_agent = DataCleanerAgent()
    feature_engineer_agent = FeatureEngineerAgent()
    
    def run_profiler(state: PipelineState):
        updates = profiler_agent.run(state)
        updates["current_phase"] = "profiling"
        return updates

    def run_cleaner_plan(state: PipelineState):
        updates = cleaner_agent.generate_plan(state)
        updates["current_phase"] = "cleaner_planning"
        return updates

    def run_cleaner_execute(state: PipelineState):
        updates = cleaner_agent.execute_plan(state)
        updates["current_phase"] = "cleaner_executing"
        return updates

    def run_engineer_plan(state: PipelineState):
        updates = feature_engineer_agent.generate_feature_plan(state)
        updates["current_phase"] = "engineer_planning"
        return updates

    def run_engineer_execute(state: PipelineState):
        updates = feature_engineer_agent.execute_plan(state)
        updates["current_phase"] = "engineer_executing"
        return updates

    workflow.add_node("profiler", run_profiler)
    workflow.add_node("cleaner_plan", run_cleaner_plan)
    workflow.add_node("cleaner_execute", run_cleaner_execute)
    workflow.add_node("engineer_plan", run_engineer_plan)
    workflow.add_node("engineer_execute", run_engineer_execute)
    
    workflow.set_entry_point("profiler")
    workflow.add_edge("profiler", "cleaner_plan")
    
    workflow.add_edge("cleaner_plan", "cleaner_execute")
    workflow.add_edge("cleaner_execute", "engineer_plan")
    workflow.add_edge("engineer_plan", "engineer_execute")
    workflow.add_edge("engineer_execute", END)
    
    memory = MemorySaver()
    app = workflow.compile(
        checkpointer=memory,
        interrupt_before=["cleaner_execute", "engineer_execute"]
    )
    
    return app
