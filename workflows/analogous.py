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
            examples = specs.get('examples', [])
            
            current_assignment_index = ctx.state.current_question_index
            
            last_answer = ctx.state.answers[-1] if ctx.state.answers else {}
            
            if not ctx.state.custom_data.get('topic'):
                return """Hello! Which topic would you like to practice today?
For example: business, job interviews, hobbies, travel, daily life, or something else?"""
            
            topic = ctx.state.custom_data.get('topic', '')
            
            if last_answer and not last_answer.get('graded'):
                student_answer = last_answer.get('answer', '')
                
                return f"""You are an English tutor checking the student's answer.

# Learning Goal
{learning_goal}

# Topic
{topic}

# Student's Answer
{student_answer}

Check the answer:
- If correct: ✅ Correct
- If incorrect: ❌ show their answer, correct version, and explanation

After feedback, ask: "Would you like to continue with another assignment?"

Be supportive."""
            else:
                examples_text = "\n".join([f"Example {i+1}: {ex}" for i, ex in enumerate(examples)])
                
                return f"""You are an English tutor.

# Learning Goal
{learning_goal}

# Flexible Part
{flexible_part}

# Topic (chosen by student)
{topic}

# Examples
{examples_text}

Create a NEW assignment on the topic "{topic}" following the format and learning goal.
Present it clearly.
Wait for the student's complete answer.

Assignment #{current_assignment_index + 1}"""
        
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
            
            if not state.custom_data.get('topic'):
                state.custom_data['topic'] = user_message
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
            
            if last_answer and not last_answer.get('graded'):
                if user_message.lower() in ['yes', 'y', 'так', 'так']:
                    state.custom_data['topic'] = ''
                    state.current_question_index += 1
                    await xano.save_workflow_state(state)
                    
                    return "Great! Which topic would you like to practice next?"
                
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
                
                if evaluation['correct']:
                    state.current_question_index += 1
                    state.custom_data['topic'] = ''
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
                    if ans.get('graded'):
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