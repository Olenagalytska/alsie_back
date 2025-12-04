from typing import Dict, List, Any
from datetime import datetime

from agents import Agent, Runner, ModelSettings, RunContextWrapper, trace

from .base import BaseWorkflow, WorkflowContext, WorkflowState, EvaluationContext


class CustomWorkflow(BaseWorkflow):
    
    def create_assistant_agent(self, context: WorkflowContext, instructions: str, model: str) -> Agent[WorkflowContext]:
        def agent_instructions(run_context: RunContextWrapper[WorkflowContext], _agent: Agent):
            ctx = run_context.context
            
            conversation_history = ""
            if ctx.state.answers:
                conversation_history = "\n\n# CONVERSATION HISTORY (what has already been discussed):\n"
                for i, ans in enumerate(ctx.state.answers, 1):
                    conversation_history += f"\nTurn {i}:\n"
                    conversation_history += f"Student: {ans.get('user_message', '')}\n"
                    conversation_history += f"You: {ans.get('assistant_response', '')}\n"
                
                conversation_history += "\n\nIMPORTANT: Review this conversation history carefully. DO NOT repeat questions or topics already covered. Move forward naturally based on what the student already knows.\n"
            
            return f"""{instructions}

{conversation_history}

Remember:
- Be flexible and conversational, not rigid
- If the student says they understand or want to move on, progress to the next topic
- Avoid repeating the same questions
- Keep responses natural and engaging
- Follow the student's pace"""
        
        return Agent[WorkflowContext](
            name="CustomAssistant",
            instructions=agent_instructions,
            model=model,
            model_settings=ModelSettings(temperature=0.7, max_tokens=1024)
        )
    
    async def run_workflow(self, block: Dict, template: Dict, user_message: str, ub_id: int, xano) -> str:
        with trace(f"Custom-{ub_id}"):
            specifications = self.parse_specifications(block)
            state = await self.load_or_create_state(ub_id, block["id"], specifications, xano)
            
            if state.status == "finished":
                return "Чат завершено."
            
            context = WorkflowContext(state=state)
            
            instructions = block.get("int_instructions", "")
            specs_text = ""
            if specifications:
                specs_text = "\n\n# Specifications:\n"
                for spec in specifications:
                    if isinstance(spec, dict):
                        for key, value in spec.items():
                            specs_text += f"{key}: {value}\n"
                    else:
                        specs_text += f"{spec}\n"
            
            full_instructions = instructions + specs_text
            
            assistant = self.create_assistant_agent(context, full_instructions, template.get("model", "gpt-4o"))
            result = await Runner.run(assistant, user_message, context=context)
            response = result.final_output_as(str)
            
            state.answers.append({
                "user_message": user_message,
                "assistant_response": response,
                "timestamp": datetime.now().isoformat()
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
        with trace(f"CustomEval-{ub_id}"):
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
                        criteria_text += f"Summary: {crit['summary_instructions']}\n"
                    if crit.get('grading_instructions'):
                        criteria_text += f"Grading: {crit['grading_instructions']}\n"
                    criteria_text += "\n"

                conversation_text = ""
                for i, ans in enumerate(ctx.workflow_state.answers):
                    conversation_text += f"\n{'='*60}\n"
                    conversation_text += f"Exchange {i+1}:\n"
                    conversation_text += f"{'='*60}\n\n"
                    conversation_text += f"**User:** {ans.get('user_message', 'N/A')}\n"
                    conversation_text += f"**Assistant:** {ans.get('assistant_response', 'N/A')}\n\n"
                
                return f"""{ctx.eval_instructions}

# Conversation History
{conversation_text}

# Evaluation Criteria
{criteria_text}

# Your Task

Evaluate the conversation based on the criteria provided.

For each criterion:
1. Review the conversation exchanges
2. Assess how well the student met the criterion
3. Assign a grade (0 to max_points for that criterion)
4. Provide clear reasoning

Format your response as:

# Evaluation Report

## Criterion 1: [Name]
**Assessment:** [Detailed assessment]
**Grade:** X/Y points
**Reasoning:** [Why this grade was assigned]

## Criterion 2: [Name]
**Assessment:** [Detailed assessment]
**Grade:** X/Y points
**Reasoning:** [Explanation]

# Summary
**Total Score:** X/{total_max_points} points
**Overall Performance:** [Brief summary]
**Recommendations:** [Optional suggestions]"""
            
            agent = Agent[EvaluationContext](
                name="CustomEvaluator",
                instructions=agent_instructions,
                model=model,
                model_settings=ModelSettings(temperature=0.3, max_tokens=2048)
            )
            
            result = await Runner.run(agent, "", context=context)
            evaluation_text = result.final_output_as(str)
            
            if isinstance(evaluation_text, str):
                evaluation_text = evaluation_text.strip()
            
            return evaluation_text