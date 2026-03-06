from .examination import ExaminationWorkflow
from .custom import CustomWorkflow
from .roleplay import RoleplayWorkflow
from .fill_gaps import FillGapsWorkflow
from .analogous import AnalogousWorkflow
from .reflection import ReflectionWorkflow
from .agent_builder import AgentBuilderWorkflow
from .custom_with_memory import CustomWithMemoryWorkflow
from .base import BaseWorkflow, WorkflowContext

__all__ = [
    "ExaminationWorkflow",
    "CustomWorkflow",
    "RoleplayWorkflow",
    "FillGapsWorkflow",
    "AnalogousWorkflow",
    "ReflectionWorkflow",
    "AgentBuilderWorkflow",
    "CustomWithMemoryWorkflow",
    "BaseWorkflow",
    "WorkflowContext",
]

WORKFLOW_REGISTRY = {
    12: ExaminationWorkflow,
    25: CustomWorkflow,
    26: FillGapsWorkflow,
    27: RoleplayWorkflow,
    28: ReflectionWorkflow,
    29: AnalogousWorkflow,
    30: CustomWithMemoryWorkflow,  # Custom agent with MCP memory management
}

def get_workflow_class(template_id: int):
    """Get workflow class by template_id."""
    return WORKFLOW_REGISTRY.get(template_id)

def get_agent_builder_workflow():
    """Get Agent Builder workflow class for OpenAI workflows."""
    return AgentBuilderWorkflow