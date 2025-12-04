from typing import Dict, List, Any
from datetime import datetime
from pydantic import BaseModel

from agents import Agent, Runner, ModelSettings, RunContextWrapper, trace

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
    
    async def run_workflow(self, block: Dict, template: Dict, user_message: str, ub_id: int, xano) -> str:
        with trace(f"Examination-{ub_id}"):
            specifications = self.parse_specifications(block)
            state = await self.load_or_create_state(ub_id, block["id"], specifications, xano)
            
            if state.status == "finished":
                return "Іспит вже завершено."
            
            if state.current_question_index >= len(state.questions):
                state.status = "finished"
                await xano.save_workflow_state(state)
                from models import ChatStatus
                await xano.update_chat_status(ub_id, status=ChatStatus.FINISHED)
                return "Вітаю! Ви відповіли на всі питання. Іспит завершено."
            
            context = WorkflowContext(state=state)
            
            if not state.answers or state.answers[-1].get('evaluation', {}).get('complete', False):
                interviewer = self.create_interviewer_agent(context, template.get("model", "gpt-4o"))
                result = await Runner.run(interviewer, "", context=context)
                response = result.final_output_as(str)
                
                state.answers.append({
                    "question_index": state.current_question_index,
                    "answer": "",
                    "timestamp": datetime.now().isoformat(),
                    "evaluation": {}
                })
                state.follow_up_count = 0
                await xano.save_workflow_state(state)
                return response
            
            state.answers[-1]['answer'] = user_message
            state.answers[-1]['timestamp'] = datetime.now().isoformat()
            
            evaluator = self.create_evaluator_agent(context, template.get("model", "gpt-4o"))
            eval_result = await Runner.run(evaluator, "", context=context)
            evaluation = eval_result.final_output.model_dump()
            
            state.answers[-1]['evaluation'] = evaluation
            
            if evaluation['complete']:
                state.current_question_index += 1
                state.follow_up_count = 0
                
                if state.current_question_index >= len(state.questions):
                    state.status = "finished"
                    await xano.save_workflow_state(state)
                    from models import ChatStatus
                    await xano.update_chat_status(ub_id, status=ChatStatus.FINISHED)
                    return "Вітаю! Ви відповіді на всі питання. Іспит завершено."
                
                await xano.save_workflow_state(state)
                
                interviewer = self.create_interviewer_agent(context, template.get("model", "gpt-4o"))
                result = await Runner.run(interviewer, "", context=context)
                response = result.final_output_as(str)
                
                state.answers.append({
                    "question_index": state.current_question_index,
                    "answer": "",
                    "timestamp": datetime.now().isoformat(),
                    "evaluation": {}
                })
                await xano.save_workflow_state(state)
                return response
            
            else:
                if state.follow_up_count >= state.max_follow_ups:
                    state.current_question_index += 1
                    state.follow_up_count = 0
                    
                    if state.current_question_index >= len(state.questions):
                        state.status = "finished"
                        await xano.save_workflow_state(state)
                        from models import ChatStatus
                        await xano.update_chat_status(ub_id, status=ChatStatus.FINISHED)
                        return "Іспит завершено."
                    
                    await xano.save_workflow_state(state)
                    
                    interviewer = self.create_interviewer_agent(context, template.get("model", "gpt-4o"))
                    result = await Runner.run(interviewer, "", context=context)
                    response = result.final_output_as(str)
                    
                    state.answers.append({
                        "question_index": state.current_question_index,
                        "answer": "",
                        "timestamp": datetime.now().isoformat(),
                        "evaluation": {}
                    })
                    await xano.save_workflow_state(state)
                    return response
                
                else:
                    state.follow_up_count += 1
                    await xano.save_workflow_state(state)
                    
                    interviewer = self.create_interviewer_agent(context, template.get("model", "gpt-4o"))
                    result = await Runner.run(interviewer, "Student answer was incomplete. Ask a follow-up question to clarify.", context=context)
                    return result.final_output_as(str)
    
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
4. Provide clear reasoning

Format your response as:

# Evaluation Report

## Criterion 1: [Name]
**Assessment:** [Detailed assessment based on answers]
**Grade:** X/Y points
**Reasoning:** [Why this grade was assigned]

## Criterion 2: [Name]
**Assessment:** [Detailed assessment]
**Grade:** X/Y points
**Reasoning:** [Explanation]

# Summary
**Total Score:** X/{total_max_points} points
**Overall Performance:** [Brief summary]
**Recommendations:** [Optional suggestions for improvement]"""
            
            agent = Agent[EvaluationContext](
                name="ExaminationEvaluator",
                instructions=agent_instructions,
                model=model,
                model_settings=ModelSettings(temperature=0.3, max_tokens=2048)
            )
            
            result = await Runner.run(agent, "", context=context)
            evaluation_text = result.final_output_as(str)
            
            if isinstance(evaluation_text, str):
                evaluation_text = evaluation_text.strip()
            
            return evaluation_text