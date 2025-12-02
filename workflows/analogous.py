from typing import Dict, List, Any
from datetime import datetime
from pydantic import BaseModel

from agents import Agent, Runner, ModelSettings, RunContextWrapper, trace

from .base import BaseWorkflow, WorkflowContext, WorkflowState, EvaluationContext


class AnalogousWorkflow(BaseWorkflow):
    
    def create_tutor_agent(self, context: WorkflowContext, specs: Dict, model: str, last_evaluation: Dict = None) -> Agent[WorkflowContext]:
        def agent_instructions(run_context: RunContextWrapper[WorkflowContext], _agent: Agent):
            ctx = run_context.context
            
            learning_goal = specs.get('learning_goal', '')
            flexible_part = specs.get('flexible part', '')
            examples = specs.get('examples', '')
            
            current_assignment_index = ctx.state.current_question_index
            topic = ctx.state.custom_data.get('topic', '')
            
            last_answer = ctx.state.answers[-1] if ctx.state.answers else {}
            
            if last_answer and last_answer.get('graded'):
                evaluation = last_answer.get('evaluation', {})
                student_answer = last_answer.get('answer', '')
                
                feedback_parts = []
                
                if evaluation.get('correct'):
                    feedback_parts.append("✅ Excellent! All answers are correct.")
                else:
                    feedback_parts.append("Let me check your answers:\n")
                    for error in evaluation.get('errors', []):
                        feedback_parts.append(f"❌ {error}")
                    feedback_parts.append(f"\n{evaluation.get('feedback', '')}")
                
                feedback_parts.append(f"\n**Assignment #{current_assignment_index + 1}**\n")
                
                return f"""You are an English tutor providing feedback and presenting the next assignment.

# Student's Previous Answer
{student_answer}

# Evaluation Result
{chr(10).join(feedback_parts)}

Now generate and present the NEXT assignment (#{current_assignment_index + 1}) following the format.

# Learning Goal
{learning_goal}

# Assignment Format
{examples}

# Topic
{topic}

Present the new assignment clearly."""
            
            elif last_answer and not last_answer.get('graded'):
                return "Wait for the student to provide their full answer before proceeding."
            
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
                return "Assignments завершено. Дякую за роботу!"
            
            context = WorkflowContext(state=state)
            
            if not state.custom_data.get('topic_chosen'):
                state.custom_data['topic'] = user_message
                state.custom_data['topic_chosen'] = True
                
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
            
            if last_answer and not last_answer.get('graded'):
                last_answer['answer'] = user_message
                last_answer['timestamp'] = datetime.now().isoformat()
                
                evaluator = self.create_evaluator_agent(context, template.get("model", "gpt-4o"))
                eval_result = await Runner.run(evaluator, "", context=context)
                evaluation = eval_result.final_output.model_dump()
                
                last_answer['evaluation'] = evaluation
                last_answer['graded'] = True
                
                state.current_question_index += 1
                
                await xano.save_workflow_state(state)
                
                tutor = self.create_tutor_agent(context, specs, template.get("model", "gpt-4o"), evaluation)
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
    
    def _format_feedback(self, evaluation: Dict, student_answer: str) -> str:
        feedback_parts = []
        
        if evaluation.get('correct', False):
            feedback_parts.append("✅ Excellent! All answers are correct.")
        else:
            feedback_parts.append("Let me check your answers:\n")
            for error in evaluation.get('errors', []):
                feedback_parts.append(f"❌ {error}")
            feedback_parts.append(f"\n{evaluation.get('feedback', '')}")
        
        return "\n".join(feedback_parts)
    
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

                assignments_text = ""
                completed_count = 0
                correct_count = 0
                
                for i, ans in enumerate(ctx.workflow_state.answers):
                    if ans.get('answer') and ans.get('graded'):
                        completed_count += 1
                        assignments_text += f"\n{'='*60}\n"
                        assignments_text += f"Assignment {ans.get('assignment_index', i) + 1}:\n"
                        assignments_text += f"{'='*60}\n\n"
                        assignments_text += f"**Task:** {ans.get('assignment', 'N/A')}\n\n"
                        assignments_text += f"**Student Answer:** {ans.get('answer', 'N/A')}\n\n"
                        
                        evaluation = ans.get('evaluation', {})
                        if evaluation:
                            correct = evaluation.get('correct', False)
                            if correct:
                                correct_count += 1
                                assignments_text += f"**Result:** ✅ All correct\n"
                            else:
                                assignments_text += f"**Result:** ❌ Has errors\n"
                                if evaluation.get('errors'):
                                    assignments_text += f"**Errors:**\n"
                                    for error in evaluation.get('errors', []):
                                        assignments_text += f"  - {error}\n"
                            if evaluation.get('feedback'):
                                assignments_text += f"**Feedback:** {evaluation.get('feedback')}\n"
                        assignments_text += "\n"

                criteria_text = ""
                for i, crit in enumerate(ctx.criteria):
                    criteria_text += f"\n## Criterion {i+1}"
                    if crit.get('criterion_name'):
                        criteria_text += f": {crit['criterion_name']}"
                    criteria_text += f"\n**Max Points:** {crit.get('max_points', 0)}\n"
                    if crit.get('summary_instructions'):
                        criteria_text += f"**Summary Instructions:** {crit['summary_instructions']}\n"
                    if crit.get('grading_instructions'):
                        criteria_text += f"**Grading Instructions:** {crit['grading_instructions']}\n"
                
                return f"""{ctx.eval_instructions}

# Summary Statistics
- Total assignments completed: {completed_count}
- Assignments with all correct answers: {correct_count}
- Accuracy rate: {(correct_count/completed_count*100) if completed_count > 0 else 0:.1f}%

# Completed Assignments
{assignments_text}

# Evaluation Criteria
{criteria_text}

# Your Task
Based on the assignments above and the evaluation criteria, provide a comprehensive evaluation of the student's English performance.

Focus on:
1. Grammar accuracy
2. Vocabulary usage
3. Understanding of the learning goal
4. Overall progress and patterns in errors

Provide specific examples from the assignments to support your evaluation."""
            
            agent = Agent[EvaluationContext](
                name="AnalogousFullEvaluator",
                instructions=agent_instructions,
                model=model,
                model_settings=ModelSettings(temperature=0.3, max_tokens=2048)
            )
            
            result = await Runner.run(agent, "", context=context)
            
            return result.final_output_as(str)