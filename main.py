import os
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

from models import StudentMessage, AssistantResponse, ChatStatus
from xano_client import XanoClient
from workflows import get_workflow_class

load_dotenv()


class Config:
    XANO_BASE_URL = os.getenv("XANO_BASE_URL", "")
    XANO_API_KEY = os.getenv("XANO_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


app = FastAPI(title="EdTech AI Platform", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.alsie.app",
        "https://alsie.app",
        "http://localhost:3000",
        "http://localhost:8000"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

xano = XanoClient(Config.XANO_BASE_URL, Config.XANO_API_KEY)


@app.get("/")
async def root():
    return {"status": "operational", "version": "3.0.0"}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "xano_configured": bool(Config.XANO_BASE_URL),
        "openai_configured": bool(Config.OPENAI_API_KEY)
    }


@app.options("/chat/message")
async def chat_message_options():
    return {"status": "ok"}


@app.post("/chat/message")
async def process_student_message(message: StudentMessage):
    try:
        print(f"=== START: Processing message for ub_id: {message.ub_id} ===")
        
        session = await xano.get_chat_session(message.ub_id)
        block = await xano.get_block(session["block_id"])
        template_data = await xano.get_template(block["int_template_id"])
        
        if session.get("status") == "idle":
            print(f"Updating status from idle to started for ub_id: {message.ub_id}")
            await xano.update_chat_status(message.ub_id, status=ChatStatus.STARTED)
        
        template_id = block["int_template_id"]
        print(f"Template ID: {template_id}")
        
        workflow_class = get_workflow_class(template_id)
        
        if not workflow_class:
            raise HTTPException(status_code=400, detail=f"No workflow found for template {template_id}")
        
        print(f"Workflow class: {workflow_class.__name__}")
        workflow = workflow_class(Config.OPENAI_API_KEY)
        
        async def generate():
            full_response = ""
            print(f"Starting stream for ub_id: {message.ub_id}")
            chunk_count = 0
            
            async for chunk in workflow.run_workflow_stream(block, template_data, message.content, message.ub_id, xano):
                chunk_count += 1
                print(f"Chunk {chunk_count}: {chunk[:50]}..." if len(chunk) > 50 else f"Chunk {chunk_count}: {chunk}")
                full_response += chunk
                yield chunk
            
            print(f"Stream complete. Total chunks: {chunk_count}")
            print(f"Full response length: {len(full_response)}")
            print(f"Full response: {full_response[:200]}..." if len(full_response) > 200 else f"Full response: {full_response}")
            
            messages_data = await xano.get_messages(message.ub_id)
            last_air_id = messages_data[-1]["id"] if messages_data else 0
            
            try:
                await xano.save_message_pair(message.ub_id, message.content, full_response, last_air_id)
                print("Message pair saved successfully")
            except Exception as save_error:
                print(f"Error saving message pair: {save_error}")
        
        print("Returning StreamingResponse")
        return StreamingResponse(
            generate(), 
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no"
            }
        )
        
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/{ub_id}/evaluate")
async def evaluate_chat(ub_id: int):
    try:
        session = await xano.get_chat_session(ub_id)
        
        if session.get('grade'):
            return {
                "evaluation": session['grade'],
                "timestamp": datetime.now().isoformat(),
                "conversation_length": 0,
                "criteria_count": 0,
                "cached": True
            }
        
        block = await xano.get_block(session["block_id"])
        
        eval_instructions = block.get("eval_instructions")
        if not eval_instructions:
            raise HTTPException(status_code=400, detail="No evaluation instructions configured")
        
        workflow_state = await xano.get_workflow_state(ub_id)
        if not workflow_state:
            raise HTTPException(status_code=404, detail="No workflow state found")
        
        import json
        criteria = block.get("eval_crit_json", [])
        if isinstance(criteria, str):
            try:
                criteria = json.loads(criteria)
            except:
                criteria = []
        
        template_id = block["int_template_id"]
        workflow_class = get_workflow_class(template_id)
        
        if not workflow_class:
            raise HTTPException(status_code=400, detail=f"No workflow found for template {template_id}")
        
        workflow = workflow_class(Config.OPENAI_API_KEY)
        
        evaluation_text = await workflow.run_evaluation(
            ub_id=ub_id,
            workflow_state=workflow_state,
            eval_instructions=eval_instructions,
            criteria=criteria,
            model=block.get("model", "gpt-4o")
        )
        
        print(f"Saving evaluation to Xano via update_ub endpoint...")
        
        update_result = await xano.update_chat_status(ub_id, grade=evaluation_text, status=ChatStatus.FINISHED)
        
        if update_result:
            print(f"Grade saved successfully: {update_result}")
        else:
            print(f"Grade save returned empty result")
        
        return {
            "evaluation": evaluation_text,
            "timestamp": datetime.now().isoformat(),
            "conversation_length": len(workflow_state.answers),
            "criteria_count": len(criteria),
            "cached": False,
            "grade_saved": bool(update_result)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/{ub_id}/state")
async def get_workflow_state(ub_id: int):
    try:
        state = await xano.get_workflow_state(ub_id)
        if state:
            return state.model_dump()
        else:
            return {"answers": [], "status": "active", "questions": []}
    except Exception as e:
        print(f"ERROR: {e}")
        return {"answers": [], "status": "active", "questions": []}


@app.get("/chat/{ub_id}/history")
async def get_chat_history(ub_id: int):
    try:
        messages = await xano.get_messages(ub_id)
        return {"messages": messages, "count": len(messages)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)