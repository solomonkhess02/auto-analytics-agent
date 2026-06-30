"""
LangGraph orchestrator workflow definition.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from core.state import PipelineState
from agents.data_profiler import DataProfilerAgent
from agents.data_cleaner import DataCleanerAgent
from agents.feature_engineer import FeatureEngineerAgent
from agents.model_trainer import ModelTrainerAgent
from agents.evaluator import EvaluatorAgent


def build_graph():
    """Builds and returns the LangGraph workflow application."""
    
    workflow = StateGraph(PipelineState)
    
    profiler_agent = DataProfilerAgent()
    cleaner_agent = DataCleanerAgent()
    feature_engineer_agent = FeatureEngineerAgent()
    model_trainer_agent = ModelTrainerAgent()
    evaluator_agent = EvaluatorAgent()
    
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

    def run_model_trainer(state: PipelineState):
        updates = model_trainer_agent.run(state)
        updates["current_phase"] = "model_training"
        return updates

    def run_evaluator(state: PipelineState):
        updates = evaluator_agent.run(state)
        updates["current_phase"] = "model_evaluating"
        return updates

    workflow.add_node("profiler", run_profiler)
    workflow.add_node("cleaner_plan", run_cleaner_plan)
    workflow.add_node("cleaner_execute", run_cleaner_execute)
    workflow.add_node("engineer_plan", run_engineer_plan)
    workflow.add_node("engineer_execute", run_engineer_execute)
    workflow.add_node("model_trainer", run_model_trainer)
    workflow.add_node("evaluator", run_evaluator)

    def route_after(next_node: str):
        """Continue to next_node unless an agent has reported errors, in which
        case halt the pipeline instead of cascading into downstream agents."""
        def _router(state: PipelineState) -> str:
            if state.get("errors"):
                return END
            return next_node
        return _router

    workflow.set_entry_point("profiler")

    workflow.add_conditional_edges("profiler", route_after("cleaner_plan"), ["cleaner_plan", END])
    workflow.add_conditional_edges("cleaner_plan", route_after("cleaner_execute"), ["cleaner_execute", END])
    workflow.add_conditional_edges("cleaner_execute", route_after("engineer_plan"), ["engineer_plan", END])
    workflow.add_conditional_edges("engineer_plan", route_after("engineer_execute"), ["engineer_execute", END])
    workflow.add_conditional_edges("engineer_execute", route_after("model_trainer"), ["model_trainer", END])
    workflow.add_conditional_edges("model_trainer", route_after("evaluator"), ["evaluator", END])
    workflow.add_edge("evaluator", END)
    
    memory = MemorySaver()
    app = workflow.compile(
        checkpointer=memory,
        interrupt_before=["cleaner_execute", "engineer_execute"]
    )
    
    return app
