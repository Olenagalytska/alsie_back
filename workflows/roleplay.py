from typing import Dict, List, Any
from datetime import datetime

from agents import Agent, Runner, ModelSettings, RunContextWrapper, trace

from .base import BaseWorkflow, WorkflowContext, WorkflowState, EvaluationContext


class RoleplayWorkflow(BaseWorkflow):
    
    def create_roleplay_agent(self, context: WorkflowContext, specs: Dict, model: str) -> Agent[WorkflowContext]:
        def agent_instructions(run_context: RunContextWrapper[WorkflowContext], _agent: Agent):
            ctx = run_context.context
            
            goal = specs.get('goal', '')
            role = specs.get('role', '')
            student_role = specs.get('student_role', '')
            behavior = specs.get('behavior', '')
            scenario = specs.get('basic_scenario', '')
            
            turn_count = len(ctx.state.answers)
            
            return f"""You are participating in a role-play simulation.

# Your Role
{role}

# Student's Role
{student_role}

# Learning Goal
{goal}

# Scenario Flow
{scenario}

# Behavior Rules
{behavior}

Current turn: {turn_count + 1}

Stay in character. Respond naturally to the student's actions and words.
Do not break the fourth wall.
Follow the behavior rules strictly."""
        
        return Agent[WorkflowContext](
            name="RoleplayAgent",
            instructions=agent_instructions,
            model=model,
            model_settings=ModelSettings(temperature=0.8, max_tokens=1024)
        )
    
    async def run_workflow(self, block: Dict, template: Dict, user_message: str, ub_id: int, xano) -> str:
        with trace(f"Roleplay-{ub_id}"):
            specifications = self.parse_specifications(block)
            specs = specifications[0] if specifications else {}
            
            state = await self.load_or_create_state(ub_id, block["id"], specifications, xano)
            
            if state.status == "finished":
                return "Role-play завершено."
            
            context = WorkflowContext(state=state)
            
            agent = self.create_roleplay_agent(context, specs, template.get("model", "gpt-4o"))
            result = await Runner.run(agent, user_message, context=context)
            response = result.final_output_as(str)
            
            state.answers.append({
                "user_message": user_message,
                "agent_response": response,
                "timestamp": datetime.now().isoformat(),
                "turn": len(state.answers) + 1
            })
            await xano.save_workflow_state(state)
            
            finish_conditions = specs.get('finish_dialogue_conditions', '')
            if self._check_finish_conditions(state, finish_conditions):
                state.status = "finished"
                await xano.save_workflow_state(state)
                from models import ChatStatus
                await xano.update_chat_status(ub_id, status=ChatStatus.FINISHED)
                return response + "\n\n[Simulation завершено]"
            
            return response
    
    def _check_finish_conditions(self, state: WorkflowState, conditions: str) -> bool:
        if not conditions:
            return False
        
        turn_count = len(state.answers)
        
        if "turns" in conditions.lower():
            import re
            match = re.search(r'(\d+)\s*turns?', conditions.lower())
            if match:
                max_turns = int(match.group(1))
                return turn_count >= max_turns
        
        return False
    
    async def run_evaluation(
        self,
        ub_id: int,
        workflow_state: WorkflowState,
        eval_instructions: str,
        criteria: List[Dict[str, Any]],
        model: str
    ) -> str:
        with trace(f"RoleplayEval-{ub_id}"):
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
                    conversation_text += f"\n### Turn {ans.get('turn', i+1)}\n"
                    conversation_text += f"**Student:** {ans.get('user_message', 'N/A')}\n"
                    conversation_text += f"**Agent:** {ans.get('agent_response', 'N/A')}\n\n"
                
                return f"""{ctx.eval_instructions}

# Role-play Conversation
{conversation_text}

# Evaluation Criteria
{criteria_text}

Evaluate the student's performance in the role-play based on the criteria."""
            
            agent = Agent[EvaluationContext](
                name="RoleplayEvaluator",
                instructions=agent_instructions,
                model=model,
                model_settings=ModelSettings(temperature=0.3, max_tokens=2048)
            )
            
            result = await Runner.run(agent, "", context=context)
            
            return result.final_output_as(str)