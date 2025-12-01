from .examination import ExaminationWorkflow
from .custom import CustomWorkflow
from .roleplay import RoleplayWorkflow
from .fill_gaps import FillGapsWorkflow
from .analogous import AnalogousWorkflow
from .reflection import ReflectionWorkflow
from .base import BaseWorkflow, WorkflowContext

__all__ = [
    "ExaminationWorkflow",
    "CustomWorkflow",
    "RoleplayWorkflow",
    "FillGapsWorkflow",
    "AnalogousWorkflow",
    "ReflectionWorkflow",
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
}

def get_workflow_class(template_id: int):
    return WORKFLOW_REGISTRY.get(template_id)