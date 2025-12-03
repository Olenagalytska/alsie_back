from typing import Dict, List, Any
from datetime import datetime
from pydantic import BaseModel

from agents import Agent, Runner, ModelSettings, RunContextWrapper, trace

from .base import BaseWorkflow, WorkflowContext, WorkflowState, EvaluationContext


class FillGapsWorkflow(BaseWorkflow):
    
    def create_tutor_agent(self, context: WorkflowContext, specs: Dict, model: str, last_evaluation: Dict = None) -> Agent[WorkflowContext]:
        def agent_instructions(run_context: RunContextWrapper[WorkflowContext], _agent: Agent):
            ctx = run_context.context
            
            learning_goal = specs.get('Learning goal', '')
            assignment_sample = specs.get('Assignment sample', '')
            additional_info = specs.get('Additional information', '')
            
            current_assignment_index = ctx.state.current_question_index
            
            if current_assignment_index >= 10:
                return "The student has completed 10 assignments. Thank them and say the test is finished."
            
            last_answer = ctx.state.answers[-1] if ctx.state.answers else {}
            
            if last_answer and last_answer.get('graded'):
                evaluation = last_answer.get('evaluation', {})
                all_correct = evaluation.get('all_correct', False)
                errors = evaluation.get('errors', [])
                student_answer = last_answer.get('answer', '')
                
                feedback_parts = []
                
                if all_correct:
                    feedback_parts.append("✅ Excellent! All answers are correct.")
                else:
                    feedback_parts.append("Let me check your answers:\n")
                    for error in errors:
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
{assignment_sample}

# Additional Information
{additional_info}

Present the new assignment clearly with numbered gaps."""
            
            elif last_answer and not last_answer.get('graded'):
                return "Wait for the student to provide their full answer before proceeding."
            
            else:
                return f"""You are an English tutor.

# Learning Goal
{learning_goal}

# Assignment Format
{assignment_sample}

# Additional Information
{additional_info}

Generate a NEW fill-in-the-gap assignment following the format of the sample.
The assignment should be about: {additional_info[:200]}

Present it clearly with numbered gaps (1. ___, 2. ___, etc).
Make it challenging but appropriate for B2 level.

**Assignment #{current_assignment_index + 1}**"""
        
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
- Accept minor spelling mistakes if meaning is clear
- Focus on grammar and word choice correctness

Return JSON:
{{
  "all_correct": true/false,
  "errors": ["gap 1: should be X", "gap 2: should be Y", ...],
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
            
            if state.current_question_index >= 10:
                state.status = "finished"
                await xano.save_workflow_state(state)
                from models import ChatStatus
                await xano.update_chat_status(ub_id, status=ChatStatus.FINISHED)
                return "You have completed 10 assignments. Excellent work! The test is finished."
            
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
                
                state.current_question_index += 1
                
                if state.current_question_index >= 10:
                    state.status = "finished"
                    await xano.save_workflow_state(state)
                    from models import ChatStatus
                    await xano.update_chat_status(ub_id, status=ChatStatus.FINISHED)
                    
                    feedback_text = self._format_feedback(evaluation, user_message)
                    return feedback_text + "\n\n🎉 You have completed all 10 assignments. Excellent work! The test is finished."
                
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
        
        if evaluation.get('all_correct', False):
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
        with trace(f"FillGapsEval-{ub_id}"):
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
                    answer_text = ans.get('answer', '')
                    
                    if answer_text:
                        completed_count += 1
                        assignments_text += f"\n{'='*60}\n"
                        assignments_text += f"Assignment {ans.get('assignment_index', i) + 1}:\n"
                        assignments_text += f"{'='*60}\n\n"
                        assignments_text += f"**Task:** {ans.get('assignment', 'N/A')}\n\n"
                        assignments_text += f"**Student Answer:** {answer_text}\n\n"
                        
                        evaluation = ans.get('evaluation', {})
                        if evaluation:
                            all_correct = evaluation.get('all_correct', False)
                            if all_correct:
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
                        else:
                            assignments_text += f"**Result:** ⚠️ Not yet evaluated\n"
                        
                        assignments_text += "\n"
                
                if completed_count == 0:
                    return "No completed assignments found. The student hasn't provided any answers yet."

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
                name="FillGapsFullEvaluator",
                instructions=agent_instructions,
                model=model,
                model_settings=ModelSettings(temperature=0.3, max_tokens=2048)
            )
            
            result = await Runner.run(agent, "", context=context)
            
            return result.final_output_as(str)