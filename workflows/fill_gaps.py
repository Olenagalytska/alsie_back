from typing import Dict, List, Any
from datetime import datetime
from pydantic import BaseModel

from agents import Agent, Runner, ModelSettings, RunContextWrapper, trace

from .base import BaseWorkflow, WorkflowContext, WorkflowState, EvaluationContext


class FillGapsWorkflow(BaseWorkflow):
    
    def create_tutor_agent(self, context: WorkflowContext, specs: Dict, model: str) -> Agent[WorkflowContext]:
        def agent_instructions(run_context: RunContextWrapper[WorkflowContext], _agent: Agent):
            ctx = run_context.context
            
            learning_goal = specs.get('Learning goal', '')
            assignment_sample = specs.get('Assignment sample', '')
            additional_info = specs.get('Additional information', '')
            
            current_assignment_index = ctx.state.current_question_index
            
            if current_assignment_index >= 10:
                return "The student has completed enough assignments. Thank them."
            
            last_answer = ctx.state.answers[-1] if ctx.state.answers else {}
            
            if last_answer and not last_answer.get('graded'):
                student_answer = last_answer.get('answer', '')
                
                return f"""You are an English tutor checking a student's answer.

# Learning Goal
{learning_goal}

# Assignment Format Reference
{assignment_sample}

# Student's Answer
{student_answer}

Check each gap:
- If correct: mark ✅
- If incorrect: show ❌, provide correct answer and brief explanation

After feedback, ask: "Would you like to try the next assignment?"

Be supportive and educational."""
            else:
                return f"""You are an English tutor.

# Learning Goal
{learning_goal}

# Assignment Format
{assignment_sample}

# Additional Information
{additional_info}

Generate a NEW fill-in-the-gap assignment following the format of the sample.
Present it clearly.
Wait for the student's full answer before providing feedback.

Assignment #{current_assignment_index + 1}"""
        
        return Agent[WorkflowContext](
            name="FillGapsTutor",
            instructions=agent_instructions,
            model=model,
            model_settings=ModelSettings(temperature=0.7, max_tokens=1024)
        )
    
    def create_evaluator_agent(self, context: WorkflowContext, model: str) -> Agent[WorkflowContext]:
        def agent_instructions(run_context: RunContextWrapper[WorkflowContext], _agent: Agent):
            ctx = run_context.context
            
            last_answer = ctx.state.answers[-1] if ctx.state.answers else {}
            student_answer = last_answer.get('answer', '')
            assignment_text = last_answer.get('assignment', '')
            
            return f"""You are evaluating a fill-in-the-gaps English assignment.

# Assignment
{assignment_text}

# Student Answer
{student_answer}

Evaluate:
- Is the answer complete (all gaps filled)?
- Are the answers correct?

Return JSON:
{{
  "all_correct": true/false,
  "errors": ["gap 1: explanation", ...],
  "feedback": "overall feedback"
}}"""
        
        class EvalOutput(BaseModel):
            all_correct: bool
            errors: List[str]
            feedback: str
        
        return Agent[WorkflowContext](
            name="GapsEvaluator",
            instructions=agent_instructions,
            model=model,
            output_type=EvalOutput,
            model_settings=ModelSettings(temperature=0.2, max_tokens=512)
        )
    
    async def run_workflow(self, block: Dict, template: Dict, user_message: str, ub_id: int, xano) -> str:
        with trace(f"FillGaps-{ub_id}"):
            specifications = self.parse_specifications(block)
            specs = specifications[0] if specifications else {}
            
            state = await self.load_or_create_state(ub_id, block["id"], specifications, xano)
            
            if state.status == "finished":
                return "Assignments завершено. Дякую за роботу!"
            
            context = WorkflowContext(state=state)
            
            last_answer = state.answers[-1] if state.answers else {}
            
            if last_answer and not last_answer.get('graded'):
                last_answer['answer'] = user_message
                last_answer['timestamp'] = datetime.now().isoformat()
                
                evaluator = self.create_evaluator_agent(context, template.get("model", "gpt-4o"))
                eval_result = await Runner.run(evaluator, "", context=context)
                evaluation = eval_result.final_output.model_dump()
                
                last_answer['evaluation'] = evaluation
                last_answer['graded'] = True
                
                await xano.save_workflow_state(state)
                
                tutor = self.create_tutor_agent(context, specs, template.get("model", "gpt-4o"))
                result = await Runner.run(tutor, "", context=context)
                response = result.final_output_as(str)
                
                if evaluation['all_correct']:
                    state.current_question_index += 1
                    await xano.save_workflow_state(state)
                
                return response
            
            else:
                if state.current_question_index >= 10:
                    state.status = "finished"
                    await xano.save_workflow_state(state)
                    from models import ChatStatus
                    await xano.update_chat_status(ub_id, status=ChatStatus.FINISHED)
                    return "Ви завершили всі завдання. Чудова робота!"
                
                tutor = self.create_tutor_agent(context, specs, template.get("model", "gpt-4o"))
                result = await Runner.run(tutor, "", context=context)
                response = result.final_output_as(str)
                
                state.answers.append({
                    "assignment_index": state.current_question_index,
                    "assignment": response,
                    "answer": "",
                    "timestamp": datetime.now().isoformat(),
                    "graded": False
                })
                await xano.save_workflow_state(state)
                
                return response
    
    async def run_evaluation(
        self,
        ub_id: int,
        workflow_state: WorkflowState,
        eval_instructions: str,
        criteria: List[Dict[str, Any]],
        model: str
    ) -> str:
        with trace(f"FillGapsEval-{ub_id}"):
            context = EvaluationContext(
                workflow_state=workflow_state,
                eval_instructions=eval_instructions,
                criteria=criteria
            )
            
            def agent_instructions(run_context: RunContextWrapper[EvaluationContext], _agent: Agent):
                ctx = run_context.context

                criteria_text = ""
                for i, crit in enumerate(ctx.criteria):
                    criteria_text += f"\n## Criterion {i+1}"
                    if crit.get('criterion_name'):
                        criteria_text += f": {crit['criterion_name']}"
                    criteria_text += f"\nMax Points: {crit.get('max_points', 0)}\n"
                    if crit.get('summary_instructions'):
                        criteria_text += f"Summary: {crit['summary_instructions']}\n"
                    if crit.get('grading_instructions'):
                        criteria_text += f"Grading: {crit['grading_instructions']}\n"

                assignments_text = ""
                for i, ans in enumerate(ctx.workflow_state.answers):
                    if ans.get('graded'):
                        assignments_text += f"\n### Assignment {i+1}\n"
                        assignments_text += f"**Task:** {ans.get('assignment', 'N/A')[:200]}...\n"
                        assignments_text += f"**Student Answer:** {ans.get('answer', 'N/A')}\n"
                        
                        evaluation = ans.get('evaluation', {})
                        if evaluation:
                            assignments_text += f"**All Correct:** {evaluation.get('all_correct', False)}\n"
                            if evaluation.get('errors'):
                                assignments_text += f"**Errors:** {', '.join(evaluation.get('errors', []))}\n"
                        assignments_text += "\n"
                
                return f"""{ctx.eval_instructions}

# Completed Assignments
{assignments_text}

# Evaluation Criteria
{criteria_text}

Evaluate the student's English performance across all assignments."""
            
            agent = Agent[EvaluationContext](
                name="FillGapsFullEvaluator",
                instructions=agent_instructions,
                model=model,
                model_settings=ModelSettings(temperature=0.3, max_tokens=2048)
            )
            
            result = await Runner.run(agent, "", context=context)
            
            return result.final_output_as(str)