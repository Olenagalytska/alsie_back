from typing import Any, AsyncIterator
from datetime import datetime
from dataclasses import dataclass

from chatkit.server import ChatKitServer, StreamingResult
from chatkit.store import MemoryStore
from chatkit.types import (
    AssistantMessageContent,
    AssistantMessageItem,
    ThreadItemDoneEvent,
    ThreadMetadata,
    ThreadStreamEvent,
    UserMessageItem,
)

from workflows import get_workflow_class


@dataclass
class RequestContext:
    user_id: str
    ub_id: int = None
    block_id: int = None


class AlsieChatKitServer(ChatKitServer[RequestContext]):
    
    def __init__(self, openai_api_key: str, xano_client):
        store = MemoryStore()
        super().__init__(store)
        self.openai_api_key = openai_api_key
        self.xano = xano_client
    
    async def respond(
        self,
        thread: ThreadMetadata,
        input: UserMessageItem | None,
        context: RequestContext,
    ) -> AsyncIterator[ThreadStreamEvent]:
        
        user_message = ""
        if input and input.content:
            for content in input.content:
                if hasattr(content, 'text'):
                    user_message = content.text
                    break
        
        if not context.ub_id or not context.block_id:
            yield ThreadItemDoneEvent(
                item=AssistantMessageItem(
                    thread_id=thread.id,
                    id=self.store.generate_item_id("message", thread, context),
                    created_at=datetime.now(),
                    content=[AssistantMessageContent(text="Error: Missing ub_id or block_id")],
                )
            )
            return
        
        try:
            block = await self.xano.get_block(context.block_id)
            template_data = await self.xano.get_template(block["int_template_id"])
            
            template_id = block["int_template_id"]
            workflow_class = get_workflow_class(template_id)
            
            if not workflow_class:
                yield ThreadItemDoneEvent(
                    item=AssistantMessageItem(
                        thread_id=thread.id,
                        id=self.store.generate_item_id("message", thread, context),
                        created_at=datetime.now(),
                        content=[AssistantMessageContent(text=f"Error: No workflow for template {template_id}")],
                    )
                )
                return
            
            workflow = workflow_class(self.openai_api_key)
            
            full_response = ""
            async for chunk in workflow.run_workflow_stream(block, template_data, user_message, context.ub_id, self.xano):
                full_response += chunk
            
            yield ThreadItemDoneEvent(
                item=AssistantMessageItem(
                    thread_id=thread.id,
                    id=self.store.generate_item_id("message", thread, context),
                    created_at=datetime.now(),
                    content=[AssistantMessageContent(text=full_response)],
                )
            )
            
        except Exception as e:
            print(f"ChatKit respond error: {e}")
            import traceback
            traceback.print_exc()
            yield ThreadItemDoneEvent(
                item=AssistantMessageItem(
                    thread_id=thread.id,
                    id=self.store.generate_item_id("message", thread, context),
                    created_at=datetime.now(),
                    content=[AssistantMessageContent(text=f"Error: {str(e)}")],
                )
            )