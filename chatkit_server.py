from typing import Any, AsyncIterator
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path

from chatkit.server import ChatKitServer, StreamingResult
from chatkit.store import NotFoundError, Store
from chatkit.types import (
    AssistantMessageContent,
    AssistantMessageItem,
    ThreadItemDoneEvent,
    ThreadMetadata,
    ThreadStreamEvent,
    UserMessageItem,
    Attachment,
    Page,
    ThreadItem,
)

from workflows import get_workflow_class


@dataclass
class RequestContext:
    user_id: str
    ub_id: int = None
    block_id: int = None


class InMemoryStore(Store[RequestContext]):
    
    def __init__(self):
        self.threads: dict[str, ThreadMetadata] = {}
        self.items: dict[str, list[ThreadItem]] = defaultdict(list)
        self.attachments: dict[str, Attachment] = {}
    
    async def load_thread(self, thread_id: str, context: RequestContext) -> ThreadMetadata:
        if thread_id not in self.threads:
            raise NotFoundError(f"Thread {thread_id} not found")
        return self.threads[thread_id]
    
    async def save_thread(self, thread: ThreadMetadata, context: RequestContext) -> None:
        self.threads[thread.id] = thread
    
    async def load_threads(
        self, limit: int, after: str | None, order: str, context: RequestContext
    ) -> Page[ThreadMetadata]:
        threads = list(self.threads.values())
        return self._paginate(
            threads, after, limit, order, 
            sort_key=lambda t: t.created_at, 
            cursor_key=lambda t: t.id
        )
    
    async def load_thread_items(
        self, thread_id: str, after: str | None, limit: int, order: str, context: RequestContext
    ) -> Page[ThreadItem]:
        items = self.items.get(thread_id, [])
        return self._paginate(
            items, after, limit, order,
            sort_key=lambda i: i.created_at,
            cursor_key=lambda i: i.id
        )
    
    async def add_thread_item(
        self, thread_id: str, item: ThreadItem, context: RequestContext
    ) -> None:
        self.items[thread_id].append(item)
    
    async def save_item(
        self, thread_id: str, item: ThreadItem, context: RequestContext
    ) -> None:
        items = self.items[thread_id]
        for idx, existing in enumerate(items):
            if existing.id == item.id:
                items[idx] = item
                return
        items.append(item)
    
    async def load_item(
        self, thread_id: str, item_id: str, context: RequestContext
    ) -> ThreadItem:
        for item in self.items.get(thread_id, []):
            if item.id == item_id:
                return item
        raise NotFoundError(f"Item {item_id} not found in thread {thread_id}")
    
    async def delete_thread(self, thread_id: str, context: RequestContext) -> None:
        self.threads.pop(thread_id, None)
        self.items.pop(thread_id, None)
    
    async def delete_thread_item(
        self, thread_id: str, item_id: str, context: RequestContext
    ) -> None:
        self.items[thread_id] = [
            item for item in self.items.get(thread_id, []) if item.id != item_id
        ]
    
    def _paginate(self, rows: list, after: str | None, limit: int, order: str, sort_key, cursor_key):
        sorted_rows = sorted(rows, key=sort_key, reverse=order == "desc")
        start = 0
        if after:
            for idx, row in enumerate(sorted_rows):
                if cursor_key(row) == after:
                    start = idx + 1
                    break
        data = sorted_rows[start : start + limit]
        has_more = start + limit < len(sorted_rows)
        next_after = cursor_key(data[-1]) if has_more and data else None
        return Page(data=data, has_more=has_more, after=next_after)
    
    async def save_attachment(self, attachment: Attachment, context: RequestContext) -> None:
        self.attachments[attachment.id] = attachment
    
    async def load_attachment(self, attachment_id: str, context: RequestContext) -> Attachment:
        if attachment_id not in self.attachments:
            raise NotFoundError(f"Attachment {attachment_id} not found")
        return self.attachments[attachment_id]
    
    async def delete_attachment(self, attachment_id: str, context: RequestContext) -> None:
        self.attachments.pop(attachment_id, None)

class DiskFileStore:
    def __init__(self, upload_dir: str = "/tmp/chatkit_uploads"):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_store = {}
    
    async def save_file(self, file_id: str, content: bytes, metadata: dict) -> None:
        file_path = self.upload_dir / file_id
        file_path.write_bytes(content)
        self.metadata_store[file_id] = metadata
    
    async def load_file(self, file_id: str) -> tuple[bytes, dict]:
        file_path = self.upload_dir / file_id
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_id} not found")
        content = file_path.read_bytes()
        metadata = self.metadata_store.get(file_id, {})
        return content, metadata
    
    async def delete_file(self, file_id: str) -> None:
        file_path = self.upload_dir / file_id
        if file_path.exists():
            file_path.unlink()
        self.metadata_store.pop(file_id, None)


class AlsieChatKitServer(ChatKitServer[RequestContext]):
    
    def __init__(self, openai_api_key: str, xano_client):
        store = InMemoryStore()
        file_store = DiskFileStore()
        super().__init__(store, file_store)
        self.file_store = file_store
        self.openai_api_key = openai_api_key
        self.xano = xano_client
    
    async def respond(
        self,
        thread: ThreadMetadata,
        input: UserMessageItem | None,
        context: RequestContext,
    ) -> AsyncIterator[ThreadStreamEvent]:
        
        user_message = ""
        files = []
        
        if input and input.content:
            for content in input.content:
                if hasattr(content, 'text'):
                    user_message += content.text
                elif hasattr(content, 'file'):
                    files.append(content.file)
        
        if files:
            file_names = [f.name for f in files]
            user_message += f"\n\n[Attached files: {', '.join(file_names)}]"
        
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
    
    async def to_message_content(self, input):
        if hasattr(input, 'file_id'):
            content, metadata = await self.file_store.load_file(input.file_id)
            return {
                "type": "text",
                "text": f"[File: {metadata.get('name', 'unknown')}]"
            }
        raise NotImplementedError()