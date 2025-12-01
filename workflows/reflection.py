from typing import Dict, List, Any
from datetime import datetime

from agents import Agent, Runner, ModelSettings, RunContextWrapper, trace

from .base import BaseWorkflow, WorkflowContext, WorkflowState, EvaluationContext


class ReflectionWorkflow(BaseWorkflow):
    
    def create_coach_agent(self, context: WorkflowContext, specs: Dict, model: str) -> Agent[WorkflowContext]:
        def agent_instructions(run_context: RunContextWrapper[WorkflowContext], _agent: Agent):
            ctx = run_context.context
            
            goal = specs.get('goal', '')
            norms = specs.get('norms', '')
            timebox = specs.get('timebox', '')
            asf = specs.get('asf', '')
            constraints = specs.get('constraints', '')
            start_template = specs.get('start_template', '')
            
            turn_count = len(ctx.state.answers)
            
            if turn_count == 0:
                return f"""You are a reflection coach using the ASF framework.

{start_template}

# Session Goal
{goal}

# Norms
{norms}

# Timebox
{timebox}

# Constraints
{constraints}

Begin the session by explaining ASF briefly and asking the first question about Aspiration."""
            
            phase = ctx.state.custom_data.get('phase', 'aspiration')
            
            asf_dict = {}
            if isinstance(asf, str):
                import json
                try:
                    asf_dict = json.loads(asf)
                except:
                    pass
            else:
                asf_dict = asf
            
            if phase == 'aspiration':
                aspiration_text = asf_dict.get('aspiration_questions', '')
                return f"""Continue the Aspiration phase.

{aspiration_text}

Ask ONE question at a time (max 120 words).
After each response, provide a brief bullet summary (2-5 points).
Push for specifics: STAR/CAR examples.
Anchor dates and metrics."""
            
            elif phase == 'strengths':
                strengths_text = asf_dict.get('strengths_questions', '')
                return f"""Continue the Strengths phase.

{strengths_text}

Ask about their strengths and values.
Identify potential overuse risks.
One question at a time.
Brief summaries after each response."""
            
            elif phase == 'feed_forward':
                feed_forward_text = asf_dict.get('feed_forward_questions', '')
                return f"""Continue the Feed-forward phase.

{feed_forward_text}

Help them commit to ONE concrete next step.
Create an if-then plan.
Set up accountability.

After this, prepare a Reflection Canvas summary."""
            
            else:
                return "The reflection session is complete. Provide a summary."
        
        return Agent[WorkflowContext](
            name="ReflectionCoach",
            instructions=agent_instructions,
            model=model,
            model_settings=ModelSettings(temperature=0.7, max_tokens=512)
        )
    
    async def run_workflow(self, block: Dict, template: Dict, user_message: str, ub_id: int, xano) -> str:
        with trace(f"Reflection-{ub_id}"):
            specifications = self.parse_specifications(block)
            specs = specifications[0] if specifications else {}
            
            state = await self.load_or_create_state(ub_id, block["id"], specifications, xano)
            
            if state.status == "finished":
                return "Reflection session завершено."
            
            context = WorkflowContext(state=state)
            
            if not state.custom_data.get('phase'):
                state.custom_data['phase'] = 'aspiration'
            
            coach = self.create_coach_agent(context, specs, template.get("model", "gpt-4o"))
            result = await Runner.run(coach, user_message, context=context)
            response = result.final_output_as(str)
            
            state.answers.append({
                "user_message": user_message,
                "coach_response": response,
                "timestamp": datetime.now().isoformat(),
                "phase": state.custom_data.get('phase', 'aspiration')
            })
            
            self._update_phase(state)
            
            await xano.save_workflow_state(state)
            
            return response
    
    def _update_phase(self, state: WorkflowState):
        turn_count = len(state.answers)
        
        if turn_count <= 5:
            state.custom_data['phase'] = 'aspiration'
        elif turn_count <= 10:
            state.custom_data['phase'] = 'strengths'
        elif turn_count <= 15:
            state.custom_data['phase'] = 'feed_forward'
        else:
            state.custom_data['phase'] = 'complete'
            state.status = 'finished'
    
    async def run_evaluation(
        self,
        ub_id: int,
        workflow_state: WorkflowState,
        eval_instructions: str,
        criteria: List[Dict[str, Any]],
        model: str
    ) -> str:
        with trace(f"ReflectionEval-{ub_id}"):
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

                conversation_text = ""
                for i, ans in enumerate(ctx.workflow_state.answers):
                    phase = ans.get('phase', 'unknown')
                    conversation_text += f"\n### Turn {i+1} [{phase.upper()}]\n"
                    conversation_text += f"**Coachee:** {ans.get('user_message', 'N/A')}\n"
                    conversation_text += f"**Coach:** {ans.get('coach_response', 'N/A')}\n\n"
                
                return f"""{ctx.eval_instructions}

# Reflection Session
{conversation_text}

# Evaluation Criteria
{criteria_text}

Evaluate the reflection session based on ASF framework and criteria."""
            
            agent = Agent[EvaluationContext](
                name="ReflectionEvaluator",
                instructions=agent_instructions,
                model=model,
                model_settings=ModelSettings(temperature=0.3, max_tokens=2048)
            )
            
            result = await Runner.run(agent, "", context=context)
            
            return result.final_output_as(str)