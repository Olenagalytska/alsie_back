# workflows/examination.py - FIXED VERSION
# FIX: Зберігаємо interviewer_question в answers

from typing import Dict, List, Any, AsyncGenerator
from datetime import datetime
from pydantic import BaseModel

from agents import Agent, Runner, ModelSettings, RunContextWrapper, trace
from openai.types.responses import ResponseTextDeltaEvent

from .base import BaseWorkflow, WorkflowContext, WorkflowState, EvaluationContext


class ExaminationWorkflow(BaseWorkflow):
    
    def create_interviewer_agent(self, context: WorkflowContext, model: str) -> Agent[WorkflowContext]:
        def agent_instructions(run_context: RunContextWrapper[WorkflowContext], _agent: Agent):
            ctx = run_context.context
            
            if ctx.state.current_question_index >= len(ctx.state.questions):
                return "The exam is complete. Thank the student."
            
            current_q = ctx.state.questions[ctx.state.current_question_index]
            last_answer = ctx.state.answers[-1] if ctx.state.answers else {}
            evaluation = last_answer.get('evaluation', {})
            
            is_followup = evaluation.get('needs_clarification', False) and ctx.state.follow_up_count > 0
            
            if is_followup:
                return f"""You are conducting an oral exam. The student gave a partial answer.

Question: {current_q['question']}
Student's previous answer: {last_answer.get('answer', '')}

Ask a NATURAL follow-up question that:
- Encourages the student to elaborate or clarify
- Does NOT reveal the correct answer or key concepts
- Uses open-ended phrasing like:
  * "Чи можете розповісти більше про..."
  * "Що ще ви знаєте про..."
  * "Уточніть, будь ласка..."

Speak in Ukrainian. Be supportive but neutral."""
            else:
                return f"""You are an examiner conducting an oral exam.

Current question: {current_q['question']}

Ask this question clearly and directly in Ukrainian.
Do NOT give hints or reveal key concepts.
Be professional and neutral."""
        
        return Agent[WorkflowContext](
            name="Interviewer",
            instructions=agent_instructions,
            model=model,
            model_settings=ModelSettings(temperature=0.7, top_p=1, max_tokens=512)
        )
    
    def create_evaluator_agent(self, context: WorkflowContext, model: str) -> Agent[WorkflowContext]:
        def agent_instructions(run_context: RunContextWrapper[WorkflowContext], _agent: Agent):
            ctx = run_context.context
            current_q = ctx.state.questions[ctx.state.current_question_index]
            last_answer = ctx.state.answers[-1] if ctx.state.answers else {}
            
            return f"""You are an evaluator for an oral examination.

QUESTION: {current_q['question']}
KEY CONCEPTS: {current_q['key_concepts']}
STUDENT ANSWER: {last_answer.get('answer', '')}

EVALUATION RULES:
1. Check if the answer SEMANTICALLY covers the key concepts
2. Accept synonyms, paraphrases, and detailed explanations
3. Focus on MEANING, not exact wording
4. If answer is clearly wrong or irrelevant → complete=false, needs_clarification=false
5. If answer partially addresses the topic → complete=false, needs_clarification=true
6. If answer fully covers the key concept (even with different words) → complete=true

Return JSON:
{{
  "complete": true/false,
  "missing_concepts": ["concept1", ...],
  "needs_clarification": true/false
}}

Current follow-up count: {ctx.state.follow_up_count}/{ctx.state.max_follow_ups}
If follow_up_count >= max, set needs_clarification=false even if incomplete."""
        
        class EvalOutput(BaseModel):
            complete: bool
            missing_concepts: List[str]
            needs_clarification: bool
        
        return Agent[WorkflowContext](
            name="Evaluator",
            instructions=agent_instructions,
            model=model,
            output_type=EvalOutput,
            model_settings=ModelSettings(temperature=0.2, max_tokens=512)
        )
    
    async def run_workflow_stream(self, block: Dict, template: Dict, user_message: str, ub_id: int, xano) -> AsyncGenerator[str, None]:
        with trace(f"Examination-{ub_id}"):
            specifications = self.parse_specifications(block)
            state = await self.load_or_create_state(ub_id, block["id"], specifications, xano)
            
            if state.status == "finished":
                yield "Іспит вже завершено."
                return
            
            if state.current_question_index >= len(state.questions):
                state.status = "finished"
                await xano.save_workflow_state(state)
                from models import ChatStatus
                await xano.update_chat_status(ub_id, status=ChatStatus.FINISHED)
                yield "Вітаю! Ви відповіли на всі питання. Іспит завершено."
                return
            
            context = WorkflowContext(state=state)
            
            # Якщо немає відповідей АБО остання відповідь завершена - задаємо нове питання
            if not state.answers or state.answers[-1].get('evaluation', {}).get('complete', False):
                interviewer = self.create_interviewer_agent(context, template.get("model", "gpt-4o"))
                result = Runner.run_streamed(interviewer, "", context=context)
                
                full_response = ""
                async for event in result.stream_events():
                    if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                        chunk = event.data.delta
                        full_response += chunk
                        yield chunk
                
                # FIX: Зберігаємо interviewer_question разом з answer
                state.answers.append({
                    "question_index": state.current_question_index,
                    "interviewer_question": full_response,  # ← FIX: Зберігаємо AI питання!
                    "answer": "",
                    "timestamp": datetime.now().isoformat(),
                    "evaluation": {}
                })
                state.follow_up_count = 0
                await xano.save_workflow_state(state)
                return
            
            # Студент відповідає
            state.answers[-1]['answer'] = user_message
            state.answers[-1]['timestamp'] = datetime.now().isoformat()
            
            # Оцінюємо відповідь
            evaluator = self.create_evaluator_agent(context, template.get("model", "gpt-4o"))
            eval_result = await Runner.run(evaluator, "", context=context)
            evaluation = eval_result.final_output.model_dump()
            
            state.answers[-1]['evaluation'] = evaluation
            
            if evaluation['complete']:
                # Відповідь повна - переходимо до наступного питання
                state.current_question_index += 1
                state.follow_up_count = 0
                
                if state.current_question_index >= len(state.questions):
                    state.status = "finished"
                    await xano.save_workflow_state(state)
                    from models import ChatStatus
                    await xano.update_chat_status(ub_id, status=ChatStatus.FINISHED)
                    yield "Вітаю! Ви відповіли на всі питання. Іспит завершено."
                    return
                
                await xano.save_workflow_state(state)
                
                # Задаємо наступне питання
                interviewer = self.create_interviewer_agent(context, template.get("model", "gpt-4o"))
                result = Runner.run_streamed(interviewer, "", context=context)
                
                full_response = ""
                async for event in result.stream_events():
                    if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                        chunk = event.data.delta
                        full_response += chunk
                        yield chunk
                
                # FIX: Зберігаємо interviewer_question
                state.answers.append({
                    "question_index": state.current_question_index,
                    "interviewer_question": full_response,  # ← FIX!
                    "answer": "",
                    "timestamp": datetime.now().isoformat(),
                    "evaluation": {}
                })
                await xano.save_workflow_state(state)
                return
            
            else:
                # Відповідь неповна
                if state.follow_up_count >= state.max_follow_ups:
                    # Вичерпали follow-ups - переходимо далі
                    state.current_question_index += 1
                    state.follow_up_count = 0
                    
                    if state.current_question_index >= len(state.questions):
                        state.status = "finished"
                        await xano.save_workflow_state(state)
                        from models import ChatStatus
                        await xano.update_chat_status(ub_id, status=ChatStatus.FINISHED)
                        yield "Іспит завершено."
                        return
                    
                    await xano.save_workflow_state(state)
                    
                    interviewer = self.create_interviewer_agent(context, template.get("model", "gpt-4o"))
                    result = Runner.run_streamed(interviewer, "", context=context)
                    
                    full_response = ""
                    async for event in result.stream_events():
                        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                            chunk = event.data.delta
                            full_response += chunk
                            yield chunk
                    
                    # FIX: Зберігаємо interviewer_question
                    state.answers.append({
                        "question_index": state.current_question_index,
                        "interviewer_question": full_response,  # ← FIX!
                        "answer": "",
                        "timestamp": datetime.now().isoformat(),
                        "evaluation": {}
                    })
                    await xano.save_workflow_state(state)
                    return
                
                else:
                    # Задаємо follow-up питання
                    state.follow_up_count += 1
                    await xano.save_workflow_state(state)
                    
                    interviewer = self.create_interviewer_agent(context, template.get("model", "gpt-4o"))
                    result = Runner.run_streamed(interviewer, "Student answer was incomplete. Ask a follow-up question to clarify.", context=context)
                    
                    full_response = ""
                    async for event in result.stream_events():
                        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                            chunk = event.data.delta
                            full_response += chunk
                            yield chunk
                    
                    # FIX: Оновлюємо останній answer з follow-up питанням
                    state.answers[-1]['follow_up_question'] = full_response  # ← FIX!
                    await xano.save_workflow_state(state)
    
    async def run_evaluation(
        self,
        ub_id: int,
        workflow_state: WorkflowState,
        eval_instructions: str,
        criteria: List[Dict[str, Any]],
        model: str
    ) -> str:
        with trace(f"ExaminationEval-{ub_id}"):
            context = EvaluationContext(
                workflow_state=workflow_state,
                eval_instructions=eval_instructions,
                criteria=criteria
            )
            
            total_max_points = self._calculate_total_points(criteria)
            
            def agent_instructions(run_context: RunContextWrapper[EvaluationContext], _agent: Agent):
                ctx = run_context.context

                criteria_text = ""
                for i, crit in enumerate(ctx.criteria):
                    criteria_text += f"\n## Criterion {i+1}"
                    if crit.get('criterion_name'):
                        criteria_text += f": {crit['criterion_name']}"
                    criteria_text += f"\nMax Points: {crit.get('max_points', 0)}\n"
                    if crit.get('summary_instructions'):
                        criteria_text += f"Summary Instructions: {crit['summary_instructions']}\n"
                    if crit.get('grading_instructions'):
                        criteria_text += f"Grading Instructions: {crit['grading_instructions']}\n"
                    criteria_text += "\n"

                conversation_text = ""
                for i, ans in enumerate(ctx.workflow_state.answers):
                    q_index = ans.get('question_index', i)
                    
                    if q_index < len(ctx.workflow_state.questions):
                        question = ctx.workflow_state.questions[q_index]
                        
                        conversation_text += f"\n{'='*60}\n"
                        conversation_text += f"Exchange {i+1}:\n"
                        conversation_text += f"{'='*60}\n\n"
                        conversation_text += f"**Question:** {question.get('question', 'N/A')}\n"
                        # FIX: Також показуємо interviewer_question якщо є
                        if ans.get('interviewer_question'):
                            conversation_text += f"**Interviewer asked:** {ans['interviewer_question']}\n"
                        conversation_text += f"**Expected key concepts:** {question.get('key_concepts', 'N/A')}\n\n"
                        conversation_text += f"**Student answer:** {ans.get('answer', 'No answer provided')}\n\n"
                        
                        evaluation = ans.get('evaluation', {})
                        if evaluation:
                            conversation_text += f"**Workflow evaluation:**\n"
                            conversation_text += f"  - Answer was complete: {evaluation.get('complete', False)}\n"
                            if evaluation.get('missing_concepts'):
                                conversation_text += f"  - Missing concepts: {', '.join(evaluation.get('missing_concepts', []))}\n"
                            if evaluation.get('needs_clarification'):
                                conversation_text += f"  - Needed clarification: {evaluation.get('needs_clarification', False)}\n"
                        
                        conversation_text += "\n"
                
                return f"""You are an evaluation assistant for an educational platform.

{ctx.eval_instructions}

# Conversation History
{conversation_text}

# Evaluation Criteria
{criteria_text}

# Your Task

Evaluate the student's performance according to the provided criteria.

For each criterion:
1. Review the student's answers and the workflow evaluation results
2. Assess how well they met the criterion
3. Assign a grade (0 to max_points for that criterion)
4. Provide brief feedback

Total maximum points available: {total_max_points}

Format your response as a clear evaluation report with:
- Score for each criterion
- Brief justification
- Total score
- Overall feedback and recommendations"""

            evaluator = Agent[EvaluationContext](
                name="FinalEvaluator",
                instructions=agent_instructions,
                model=model,
                model_settings=ModelSettings(temperature=0.3, max_tokens=2048)
            )
            
            result = await Runner.run(evaluator, "", context=context)
            return result.final_output
    
    def _calculate_total_points(self, criteria: List[Dict[str, Any]]) -> int:
        total = 0
        for crit in criteria:
            try:
                total += int(crit.get('max_points', 0))
            except (ValueError, TypeError):
                pass
        return total