import httpx
import json
import re
from typing import Dict, Any, List, Optional
from datetime import datetime

from workflows.base import WorkflowState
from models import ChatStatus


class XanoClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.client = httpx.AsyncClient(headers=headers, timeout=30.0)
    
    async def get_block(self, block_id: int) -> Dict[str, Any]:
        response = await self.client.get(f"{self.base_url}/block/{block_id}")
        response.raise_for_status()
        return response.json()
    
    async def get_template(self, template_id: int) -> Dict[str, Any]:
        response = await self.client.get(f"{self.base_url}/template/{template_id}")
        response.raise_for_status()
        return response.json()
    
    async def get_chat_session(self, ub_id: int) -> Dict[str, Any]:
        response = await self.client.get(f"{self.base_url}/ub/{ub_id}")
        response.raise_for_status()
        return response.json()
    
    async def get_workflow_state(self, ub_id: int) -> Optional[WorkflowState]:
        try:
            response = await self.client.get(f"{self.base_url}/get_workflow_state/{ub_id}")
            if response.status_code == 200:
                data = response.json()
                if data and not data.get('error'):
                    data['questions'] = json.loads(data['questions']) if isinstance(data.get('questions'), str) else data.get('questions', [])
                    data['answers'] = json.loads(data['answers']) if isinstance(data.get('answers'), str) else data.get('answers', [])
                    data['custom_data'] = json.loads(data['custom_data']) if isinstance(data.get('custom_data'), str) else data.get('custom_data', {})
                    return WorkflowState(**data)
        except Exception as e:
            print(f"Error loading workflow state: {e}")
        return None
    
    async def save_workflow_state(self, state: WorkflowState):
        data = {
            "ub_id": state.ub_id,
            "block_id": state.block_id,
            "current_question_index": state.current_question_index,
            "questions": json.dumps(state.questions, ensure_ascii=False),
            "answers": json.dumps(state.answers, ensure_ascii=False),
            "follow_up_count": state.follow_up_count,
            "max_follow_ups": state.max_follow_ups,
            "status": state.status,
            "custom_data": json.dumps(state.custom_data, ensure_ascii=False)
        }
        response = await self.client.post(f"{self.base_url}/save_workflow_state", json=data)
        return response.json() if response.status_code in [200, 201] else None
    
    async def get_messages(self, ub_id: int) -> List[Dict[str, Any]]:
        response = await self.client.get(f"{self.base_url}/air", params={"ub_id": ub_id})
        response.raise_for_status()
        return response.json()
    
    async def save_message_pair(self, ub_id: int, user_message: str, ai_response: str, prev_id: Optional[int] = None) -> Dict[str, Any]:
        timestamp = int(datetime.now().timestamp() * 1000)
        message_record = {
            "ub_id": ub_id,
            "created_at": timestamp,
            "status": "new",
            "user_content": json.dumps({"type": "text", "text": user_message, "created_at": timestamp}),
            "ai_content": json.dumps([{"text": ai_response, "title": "", "created_at": timestamp}]),
            "prev_id": prev_id if prev_id else 0
        }
        response = await self.client.post(f"{self.base_url}/add_air", json=message_record)
        return response.json() if response.status_code in [200, 201] else {"id": timestamp}
    
    def _extract_score(self, evaluation_text: str) -> Optional[float]:
        patterns = [
            r'\*\*Total Score:\*\*\s*(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)',
            r'Total Score:\s*(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)',
            r'Загальна оцінка:\s*(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, evaluation_text)
            if match:
                try:
                    score = float(match.group(1))
                    max_score = float(match.group(2))
                    if max_score > 0:
                        return round(score, 2)
                except (ValueError, ZeroDivisionError):
                    continue
        
        return None
    
    async def update_chat_status(self, ub_id: int, status: Optional[ChatStatus] = None, grade: Optional[str] = None, last_air_id: Optional[int] = None):
        update_data = {"ub_id": int(ub_id)}
        
        if status:
            update_data["status"] = status.value
        
        if grade is not None:
            update_data["work_summary"] = grade
            
            score = self._extract_score(grade)
            if score is not None:
                update_data["grade"] = score
                print(f"Extracted score: {score}")
            else:
                print("Could not extract numerical score from evaluation")
        
        if last_air_id:
            update_data["last_air_id"] = int(last_air_id)
        
        try:
            print(f"Updating UB {ub_id} with data: {update_data}")
            response = await self.client.post(f"{self.base_url}/update_ub", json=update_data)
            if response.status_code in [200, 201]:
                result = response.json()
                print(f"Update UB successful: {result}")
                return result
            else:
                print(f"Update UB error: {response.status_code}")
                print(f"Response: {response.text[:500]}")
        except Exception as e:
            print(f"Status update error: {e}")
            import traceback
            traceback.print_exc()
        return None