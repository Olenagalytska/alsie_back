from typing import Dict, List, Any
from datetime import datetime
from pydantic import BaseModel

from agents import Agent, Runner, ModelSettings, RunContextWrapper, trace

from .base import BaseWorkflow, WorkflowContext, WorkflowState, EvaluationContext


class AnalogousWorkflow(BaseWorkflow):
    
    def create_tutor_agent(self, context: WorkflowContext, specs: Dict, model: str) -> Agent[WorkflowContext]:
        def agent_instructions(run_context: RunContextWrapper[WorkflowContext], _agent: Agent):
            ctx = run_context.context
            
            learning_goal = specs.get('learning_goal', '')
            flexible_part = specs.get('flexible part', '')
            examples = specs.get('examples', '')
            
            current_assignment_index = ctx.state.current_question_index
            
            last_answer = ctx.state.answers[-1] if ctx.state.answers else {}
            
            if not ctx.state.custom_data.get('topic_chosen'):
                return """Hello! Which topic would you like to practice today?
For example: business, job interviews, hobbies, travel, daily life, or something else?"""
            
            topic = ctx.state.custom_data.get('topic', '')
            
            if last_answer and last_answer.get('graded'):
                evaluation = last_answer.get('evaluation', {})
                student_answer = last_answer.get('answer', '')
                
                feedback_parts = []
                
                if evaluation.get('correct'):
                    feedback_parts.append("✅ Correct! Well done.")
                else:
                    feedback_parts.append("Let me check your answer:\n")
                    for error in evaluation.get('errors', []):
                        feedback_parts.append(f"❌ {error}")
                    if evaluation.get('feedback'):
                        feedback_parts.append(f"\n{evaluation['feedback']}")
                
                feedback_parts.append("\n\nWould you like to continue with another assignment?")
                
                return "\n".join(feedback_parts)
            
            elif last_answer and not last_answer.get('graded') and not last_answer.get('answer'):
                return f"""You are an English tutor waiting for the student's answer.

The student was given this assignment:
{last_answer.get('assignment', '')}

They haven't provided their answer yet. Wait for their response.
Do NOT generate a new assignment.
Just wait for the student to provide their complete answer."""
            
            else:
                return f"""You are an English tutor.

# Learning Goal
{learning_goal}

# Flexible Part
{flexible_part}

# Topic (chosen by student)
{topic}

# Examples (format reference)
{examples}

Create a NEW assignment on the topic "{topic}" following the format and learning goal from the examples.
Present it clearly.
Wait for the student's complete answer.

**Assignment #{current_assignment_index + 1}**"""
        
        return Agent[WorkflowContext](
            name="AnalogousTutor",
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
            
            return f"""Evaluate the English assignment answer.

# Assignment
{assignment_text}

# Student Answer
{student_answer}

Check:
- Is the answer complete?
- Are grammar and vocabulary correct?
- Does it address the task?

Return JSON:
{{
  "correct": true/false,
  "errors": ["error explanation", ...],
  "feedback": "overall feedback"
}}"""
        
        class EvalOutput(BaseModel):
            correct: bool
            errors: List[str]
            feedback: str
        
        return Agent[WorkflowContext](
            name="AnalogousEvaluator",
            instructions=agent_instructions,
            model=model,
            output_type=EvalOutput,
            model_settings=ModelSettings(temperature=0.2, max_tokens=512)
        )
    
    async def run_workflow(self, block: Dict, template: Dict, user_message: str, ub_id: int, xano) -> str:
        with trace(f"Analogous-{ub_id}"):
            specifications = self.parse_specifications(block)
            specs = specifications[0] if specifications else {}
            
            state = await self.load_or_create_state(ub_id, block["id"], specifications, xano)
            
            if state.status == "finished":
                return "Assignments завершено. Дякую!"
            
            context = WorkflowContext(state=state)
            
            # Step 1: Choose topic
            if not state.custom_data.get('topic_chosen'):
                state.custom_data['topic'] = user_message
                state.custom_data['topic_chosen'] = True
                await xano.save_workflow_state(state)
                
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
            
            last_answer = state.answers[-1] if state.answers else {}
            
            # Step 2: Student provides answer
            if last_answer and not last_answer.get('graded') and not last_answer.get('answer'):
                # Check if student wants to continue instead of answering
                if user_message.lower() in ['yes', 'y', 'так', 'continue', 'next']:
                    state.custom_data['topic_chosen'] = False
                    state.custom_data['topic'] = ''
                    state.current_question_index += 1
                    await xano.save_workflow_state(state)
                    
                    return "Great! Which topic would you like to practice next?"
                
                # Save student's answer
                last_answer['answer'] = user_message
                last_answer['timestamp'] = datetime.now().isoformat()
                await xano.save_workflow_state(state)
                
                # Evaluate
                evaluator = self.create_evaluator_agent(context, template.get("model", "gpt-4o"))
                eval_result = await Runner.run(evaluator, "", context=context)
                evaluation = eval_result.final_output.model_dump()
                
                last_answer['evaluation'] = evaluation
                last_answer['graded'] = True
                await xano.save_workflow_state(state)
                
                # Give feedback
                tutor = self.create_tutor_agent(context, specs, template.get("model", "gpt-4o"))
                result = await Runner.run(tutor, "", context=context)
                response = result.final_output_as(str)
                
                return response
            
            # Step 3: After feedback - continue or finish
            elif last_answer and last_answer.get('graded'):
                if user_message.lower() in ['yes', 'y', 'так', 'continue']:
                    state.custom_data['topic_chosen'] = False
                    state.custom_data['topic'] = ''
                    state.current_question_index += 1
                    await xano.save_workflow_state(state)
                    
                    return "Great! Which topic would you like to practice next?"
                elif user_message.lower() in ['no', 'n', 'ні', 'finish', 'stop']:
                    state.status = "finished"
                    await xano.save_workflow_state(state)
                    from models import ChatStatus
                    await xano.update_chat_status(ub_id, status=ChatStatus.FINISHED)
                    return "Thank you for practicing! Great work today. 🎉"
                else:
                    return "Would you like to continue with another assignment? (yes/no)"
            
            # Fallback: generate new assignment
            else:
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
        with trace(f"AnalogousEval-{ub_id}"):
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
                    if ans.get('graded') and ans.get('answer'):
                        assignments_text += f"\n### Assignment {i+1}\n"
                        assignments_text += f"**Task:** {ans.get('assignment', 'N/A')[:200]}...\n"
                        assignments_text += f"**Student Answer:** {ans.get('answer', 'N/A')}\n"
                        
                        evaluation = ans.get('evaluation', {})
                        if evaluation:
                            assignments_text += f"**Correct:** {evaluation.get('correct', False)}\n"
                            if evaluation.get('errors'):
                                assignments_text += f"**Errors:** {', '.join(evaluation.get('errors', []))}\n"
                        assignments_text += "\n"
                
                return f"""{ctx.eval_instructions}

# Completed Assignments
{assignments_text}

# Evaluation Criteria
{criteria_text}

Evaluate the student's English performance."""
            
            agent = Agent[EvaluationContext](
                name="AnalogousFullEvaluator",
                instructions=agent_instructions,
                model=model,
                model_settings=ModelSettings(temperature=0.3, max_tokens=2048)
            )
            
            result = await Runner.run(agent, "", context=context)
            
            return result.final_output_as(str)