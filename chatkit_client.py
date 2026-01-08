import httpx
from typing import Optional, Dict, Any, AsyncIterator
from dataclasses import dataclass


@dataclass
class ChatKitSession:
    session_id: str
    client_secret: str
    workflow_id: str
    expires_at: int
    user: str


class ChatKitClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1/chatkit"
        self.client = httpx.AsyncClient(timeout=60.0)
    
    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "chatkit_beta=v1"
        }
    
    async def create_session(
        self, 
        workflow_id: str, 
        user_id: str,
        expires_after: int = 3600,
        max_requests_per_minute: int = 30
    ) -> ChatKitSession:
        response = await self.client.post(
            f"{self.base_url}/sessions",
            headers=self._headers(),
            json={
                "workflow": {"id": workflow_id},
                "user": user_id,
                "expires_after": expires_after,
                "max_requests_per_1_minute": max_requests_per_minute
            }
        )
        response.raise_for_status()
        data = response.json()
        
        return ChatKitSession(
            session_id=data["id"],
            client_secret=data["client_secret"],
            workflow_id=data["workflow"]["id"],
            expires_at=data["expires_at"],
            user=data["user"]
        )
    
    async def list_threads(
        self, 
        user_id: Optional[str] = None,
        limit: int = 20
    ) -> Dict[str, Any]:
        params = {"limit": limit}
        if user_id:
            params["user"] = user_id
        
        response = await self.client.get(
            f"{self.base_url}/threads",
            headers=self._headers(),
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    async def get_thread(self, thread_id: str) -> Dict[str, Any]:
        response = await self.client.get(
            f"{self.base_url}/threads/{thread_id}",
            headers=self._headers()
        )
        response.raise_for_status()
        return response.json()
    
    async def get_thread_items(
        self, 
        thread_id: str,
        limit: int = 100
    ) -> Dict[str, Any]:
        response = await self.client.get(
            f"{self.base_url}/threads/{thread_id}/items",
            headers=self._headers(),
            params={"limit": limit}
        )
        response.raise_for_status()
        return response.json()
    
    async def delete_thread(self, thread_id: str) -> Dict[str, Any]:
        response = await self.client.delete(
            f"{self.base_url}/threads/{thread_id}",
            headers=self._headers()
        )
        response.raise_for_status()
        return response.json()
    
    async def cancel_session(self, session_id: str) -> Dict[str, Any]:
        response = await self.client.post(
            f"{self.base_url}/sessions/{session_id}/cancel",
            headers=self._headers()
        )
        response.raise_for_status()
        return response.json()
    
    def extract_messages_from_thread(self, thread_data: Dict[str, Any]) -> list:
        messages = []
        items = thread_data.get("items", {}).get("data", [])
        
        for item in items:
            item_type = item.get("type")
            content = item.get("content", [])
            
            if item_type == "user_message":
                for c in content:
                    if c.get("type") == "input_text":
                        messages.append({
                            "role": "user",
                            "content": c.get("text", "")
                        })
            
            elif item_type == "assistant_message":
                for c in content:
                    if c.get("type") == "output_text":
                        messages.append({
                            "role": "assistant", 
                            "content": c.get("text", "")
                        })
        
        return messages
    
    async def close(self):
        await self.client.aclose()