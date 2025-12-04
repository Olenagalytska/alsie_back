from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from datetime import datetime
import json


class WorkflowState(BaseModel):
    ub_id: int
    block_id: int
    current_question_index: int = 0
    questions: List[Dict[str, Any]] = []
    answers: List[Dict[str, Any]] = []
    follow_up_count: int = 0
    max_follow_ups: int = 3
    status: str = "active"
    custom_data: Dict[str, Any] = {}


class WorkflowContext:
    def __init__(self, state: WorkflowState):
        self.state = state


class EvaluationContext:
    def __init__(
        self,
        workflow_state: WorkflowState,
        eval_instructions: str,
        criteria: List[Dict[str, Any]]
    ):
        self.workflow_state = workflow_state
        self.eval_instructions = eval_instructions
        self.criteria = criteria


class BaseWorkflow(ABC):
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
    
    @abstractmethod
    async def run_workflow(
        self,
        block: Dict,
        template: Dict,
        user_message: str,
        ub_id: int,
        xano
    ) -> str:
        pass
    
    @abstractmethod
    async def run_evaluation(
        self,
        ub_id: int,
        workflow_state: WorkflowState,
        eval_instructions: str,
        criteria: List[Dict[str, Any]],
        model: str
    ) -> str:
        pass
    
    async def load_or_create_state(
        self,
        ub_id: int,
        block_id: int,
        specifications: List[Dict],
        xano
    ) -> WorkflowState:
        state = await xano.get_workflow_state(ub_id)
        
        if not state:
            state = WorkflowState(
                ub_id=ub_id,
                block_id=block_id,
                questions=specifications,
                current_question_index=0,
                answers=[],
                follow_up_count=0,
                status="active"
            )
            await xano.save_workflow_state(state)
        
        return state
    
    def parse_specifications(self, block: Dict) -> List[Dict]:
        specifications = block.get("specifications", [])
        if isinstance(specifications, str):
            try:
                specifications = json.loads(specifications)
            except:
                specifications = []
        return specifications
    
    def parse_criteria(self, block: Dict) -> List[Dict]:
        criteria = block.get("eval_crit_json", [])
        if isinstance(criteria, str):
            try:
                criteria = json.loads(criteria)
            except:
                criteria = []
        return criteria
    
    def _calculate_total_points(self, criteria: List[Dict[str, Any]]) -> int:
        total = 0
        for crit in criteria:
            total += crit.get('max_points', 0)
        return total
    
    def _append_score_summary(self, evaluation_text: str, criteria: List[Dict[str, Any]]) -> str:
        return evaluation_text