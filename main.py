import os
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from models import StudentMessage, AssistantResponse, ChatStatus
from xano_client import XanoClient
from chatkit_client import ChatKitClient
from workflows import get_workflow_class

load_dotenv()


class Config:
    XANO_BASE_URL = os.getenv("XANO_BASE_URL", "")
    XANO_API_KEY = os.getenv("XANO_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


app = FastAPI(title="EdTech AI Platform", version="4.0.1")

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
chatkit = ChatKitClient(Config.OPENAI_API_KEY)


class ChatKitSessionRequest(BaseModel):
    ub_id: int
    user_id: str


class ChatKitSessionResponse(BaseModel):
    session_id: str
    client_secret: str
    workflow_id: str
    expires_at: int


@app.get("/")
async def root():
    return {"status": "operational", "version": "4.0.1", "chatkit_enabled": True}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "xano_configured": bool(Config.XANO_BASE_URL),
        "openai_configured": bool(Config.OPENAI_API_KEY),
        "chatkit_enabled": True
    }


@app.options("/chat/message")
async def chat_message_options():
    return {"status": "ok"}


@app.options("/chatkit/session")
async def chatkit_session_options():
    return {"status": "ok"}


@app.post("/chatkit/session", response_model=ChatKitSessionResponse)
async def create_chatkit_session(request: ChatKitSessionRequest):
    try:
        session = await xano.get_chat_session(request.ub_id)
        block = await xano.get_block(session["block_id"])
        
        workflow_id = block.get("workflow_id")
        if not workflow_id:
            raise HTTPException(
                status_code=400, 
                detail="Block does not have a workflow_id configured"
            )
        
        chatkit_session = await chatkit.create_session(
            workflow_id=workflow_id,
            user_id=request.user_id
        )
        
        return ChatKitSessionResponse(
            session_id=chatkit_session.session_id,
            client_secret=chatkit_session.client_secret,
            workflow_id=chatkit_session.workflow_id,
            expires_at=chatkit_session.expires_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR creating ChatKit session: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chatkit/check/{ub_id}")
async def check_chatkit_enabled(ub_id: int):
    try:
        session = await xano.get_chat_session(ub_id)
        block = await xano.get_block(session["block_id"])
        
        workflow_id = block.get("workflow_id")
        
        return {
            "ub_id": ub_id,
            "chatkit_enabled": bool(workflow_id),
            "workflow_id": workflow_id
        }
        
    except Exception as e:
        print(f"ERROR checking ChatKit: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/message")
async def process_student_message(message: StudentMessage):
    try:
        session = await xano.get_chat_session(message.ub_id)
        block = await xano.get_block(session["block_id"])
        
        # Якщо є workflow_id - повідомити що потрібно використовувати ChatKit
        if block.get("workflow_id"):
            raise HTTPException(
                status_code=400, 
                detail="This block uses ChatKit. Use /chatkit/session endpoint instead."
            )
        
        template_data = await xano.get_template(block["int_template_id"])
        
        template_id = block["int_template_id"]
        workflow_class = get_workflow_class(template_id)
        
        if not workflow_class:
            raise HTTPException(status_code=400, detail=f"No workflow found for template {template_id}")
        
        workflow = workflow_class(Config.OPENAI_API_KEY)
        
        async def generate():
            full_response = ""
            async for chunk in workflow.run_workflow_stream(block, template_data, message.content, message.ub_id, xano):
                full_response += chunk
                yield chunk
            
            messages_data = await xano.get_messages(message.ub_id)
            last_air_id = messages_data[-1]["id"] if messages_data else 0
            await xano.save_message_pair(message.ub_id, message.content, full_response, last_air_id)
        
        return StreamingResponse(
            generate(), 
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/{ub_id}/evaluate")
async def evaluate_chat(ub_id: int):
    try:
        session = await xano.get_chat_session(ub_id)
        block = await xano.get_block(session["block_id"])
        
        if session.get('grade'):
            return {
                "evaluation": session['grade'],
                "timestamp": datetime.now().isoformat(),
                "conversation_length": 0,
                "criteria_count": 0,
                "cached": True
            }
        
        eval_instructions = block.get("eval_instructions")
        if not eval_instructions:
            raise HTTPException(status_code=400, detail="No evaluation instructions configured")
        
        import json
        criteria = block.get("eval_crit_json", [])
        if isinstance(criteria, str):
            try:
                criteria = json.loads(criteria)
            except:
                criteria = []
        
        if block.get("workflow_id"):
            evaluation_text = await evaluate_chatkit_conversation(ub_id, block, eval_instructions, criteria)
        else:
            workflow_state = await xano.get_workflow_state(ub_id)
            if not workflow_state:
                raise HTTPException(status_code=404, detail="No workflow state found")
            
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


async def evaluate_chatkit_conversation(
    ub_id: int, 
    block: dict, 
    eval_instructions: str, 
    criteria: list
) -> str:
    from agents import Agent, Runner, ModelSettings
    
    messages = await xano.get_messages(ub_id)
    
    if not messages:
        return "No conversation found to evaluate."
    
    conversation_text = ""
    for msg in messages:
        try:
            import json
            user_content = msg.get("user_content", "{}")
            if isinstance(user_content, str):
                user_content = json.loads(user_content)
            user_text = user_content.get("text", "")
            
            ai_content = msg.get("ai_content", "[]")
            if isinstance(ai_content, str):
                ai_content = json.loads(ai_content)
            ai_text = ai_content[0].get("text", "") if ai_content else ""
            
            if user_text:
                conversation_text += f"\n**Student:** {user_text}\n"
            if ai_text:
                conversation_text += f"**Assistant:** {ai_text}\n"
        except Exception as e:
            print(f"Error parsing message: {e}")
            continue
    
    criteria_text = ""
    for i, crit in enumerate(criteria, 1):
        criteria_text += f"\n## Criterion {i}"
        if crit.get('criterion_name'):
            criteria_text += f": {crit['criterion_name']}"
        criteria_text += f"\nMax Points: {crit.get('max_points', 0)}\n"
        if crit.get('summary_instructions'):
            criteria_text += f"Summary Instructions: {crit['summary_instructions']}\n"
        if crit.get('grading_instructions'):
            criteria_text += f"Grading Instructions: {crit['grading_instructions']}\n"
    
    eval_agent = Agent(
        name="EvaluationAgent",
        instructions=f"""{eval_instructions}

# Conversation to Evaluate
{conversation_text}

# Evaluation Criteria
{criteria_text}

# Your Task
Evaluate the student's performance according to the provided criteria.

For each criterion:
1. Write **Assessment:** describing how well the student met this criterion
2. Write **Grade:** X/Y points
3. Write **Reasoning:** explaining why you gave this grade

At the end, provide:
# Summary
**Total Score:** X/Y points
**Overall Performance:** Brief summary
**Recommendations:** Suggestions for improvement
""",
        model=block.get("model", "gpt-4o"),
        model_settings=ModelSettings(temperature=0.3)
    )
    
    result = await Runner.run(eval_agent, "Please evaluate the conversation above.")
    return result.final_output


@app.get("/chat/{ub_id}/state")
async def get_workflow_state(ub_id: int):
    try:
        state = await xano.get_workflow_state(ub_id)
        if state:
            return state.model_dump()
        else:
            raise HTTPException(status_code=404, detail="Workflow state not found")
    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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