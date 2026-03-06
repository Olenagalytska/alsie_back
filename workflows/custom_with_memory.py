from typing import Dict, List, Any, AsyncGenerator
from datetime import datetime

from agents import Agent, Runner, ModelSettings, RunContextWrapper, trace
from openai.types.responses import ResponseTextDeltaEvent

from .base import BaseWorkflow, WorkflowContext, WorkflowState, EvaluationContext


class CustomWithMemoryWorkflow(BaseWorkflow):
    """
    Workflow for a custom agent with user memory management via an MCP server.

    Mirrors the OpenAI workflow builder export (custom_workflow_with_memory).
    The agent can read and write user data through a Xano-hosted MCP server
    (get_user_data / write_user_data tools), enabling persistent memory across
    sessions at course, lesson, or block level.

    Expected block fields:
        mcp_server_url              - Xano MCP streaming endpoint URL
        int_instructions            - main agent instructions
        specifications              - agent specifications (parsed by base)
        level                       - "course" | "lesson" | "block"
        include_all_children        - bool, whether to fetch child records
        reading_user_data_instructions  - how the agent should read user data
        writing_user_data_instructions  - how the agent should write user data
    """

    def _make_instructions_fn(
        self,
        course_id: Any,
        lesson_id: Any,
        block_id: Any,
        user_id: Any,
        include_all_children: Any,
        level: str,
        reading_instructions: str,
        writing_instructions: str,
        agent_instructions: str,
        specs_text: str,
    ):
        """Return a dynamic instruction function compatible with the agents SDK."""

        def instructions_fn(run_context: RunContextWrapper[WorkflowContext], _agent: Agent) -> str:
            return (
                f"Use these inputs to call the MCP server when needed.\n"
                f"course_id: {course_id}\n"
                f"lesson_id: {lesson_id}\n"
                f"block_id: {block_id}\n"
                f"user_id: {user_id}\n"
                f"include_all_children: {str(include_all_children).lower()}\n"
                f"level: {level}\n\n"
                f"# Reading user data\n{reading_instructions}\n\n"
                f"# Writing data about user\n{writing_instructions}\n\n"
                f"# Instructions\n{agent_instructions}\n\n"
                f"# Specifications\n{specs_text}"
            )

        return instructions_fn

    async def run_workflow_stream(
        self,
        block: Dict,
        template: Dict,
        user_message: str,
        ub_id: int,
        xano,
    ) -> AsyncGenerator[str, None]:
        with trace(f"CustomWithMemory-{ub_id}"):
            specifications = self.parse_specifications(block)
            state = await self.load_or_create_state(ub_id, block["id"], specifications, xano)

            if state.status == "finished":
                yield "Чат завершено."
                return

            # Fetch user_id from session (not available on block directly)
            session = await xano.get_chat_session(ub_id)
            user_id = session.get("user_id", 0)

            # IDs for MCP context
            block_id = block.get("id", 0)
            lesson = block.get("_lesson") or {}
            lesson_id = lesson.get("id", 0)
            course_id = (
                lesson.get("course_id")
                or (lesson.get("_course") or {}).get("id", 0)
                or 0
            )

            # Memory / MCP config from block
            mcp_server_url = block.get("mcp_server_url", "")
            level = block.get("level", "course")
            include_all_children = block.get("include_all_children", True)
            reading_instructions = block.get(
                "reading_user_data_instructions",
                "Call MCP server to get user data.",
            )
            writing_instructions = block.get(
                "writing_user_data_instructions",
                "When the user mentions a general fact about themselves, write the information.",
            )
            agent_instructions_text = block.get("int_instructions", "You are a helpful agent.")

            # Build specifications text
            specs_text = ""
            for spec in specifications:
                if isinstance(spec, dict):
                    for key, value in spec.items():
                        specs_text += f"{key}: {value}\n"
                else:
                    specs_text += f"{spec}\n"

            instructions_fn = self._make_instructions_fn(
                course_id=course_id,
                lesson_id=lesson_id,
                block_id=block_id,
                user_id=user_id,
                include_all_children=include_all_children,
                level=level,
                reading_instructions=reading_instructions,
                writing_instructions=writing_instructions,
                agent_instructions=agent_instructions_text,
                specs_text=specs_text,
            )

            model = template.get("model", "gpt-4o")

            # Reconstruct conversation history for proper multi-turn context
            history = state.custom_data.get("conversation_history", [])
            input_messages = [*history, {"role": "user", "content": user_message}]

            context = WorkflowContext(state=state)
            full_response = ""

            if mcp_server_url:
                from agents.mcp import MCPServerSse

                async with MCPServerSse(
                    name="user_data",
                    params={"url": mcp_server_url},
                ) as mcp_server:
                    agent = Agent[WorkflowContext](
                        name="CustomAgentWithMemory",
                        instructions=instructions_fn,
                        model=model,
                        mcp_servers=[mcp_server],
                        model_settings=ModelSettings(temperature=0.7, max_tokens=1024),
                    )
                    result = Runner.run_streamed(agent, input_messages, context=context)
                    async for event in result.stream_events():
                        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                            chunk = event.data.delta
                            full_response += chunk
                            yield chunk
            else:
                # Fallback: run without MCP tools when no server URL is configured
                agent = Agent[WorkflowContext](
                    name="CustomAgentWithMemory",
                    instructions=instructions_fn,
                    model=model,
                    model_settings=ModelSettings(temperature=0.7, max_tokens=1024),
                )
                result = Runner.run_streamed(agent, input_messages, context=context)
                async for event in result.stream_events():
                    if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                        chunk = event.data.delta
                        full_response += chunk
                        yield chunk

            # Persist conversation history for next turn
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": full_response})
            state.custom_data["conversation_history"] = history

            state.answers.append({
                "user_message": user_message,
                "assistant_response": full_response,
                "timestamp": datetime.now().isoformat(),
            })
            await xano.save_workflow_state(state)

    async def run_evaluation(
        self,
        ub_id: int,
        workflow_state: WorkflowState,
        eval_instructions: str,
        criteria: List[Dict[str, Any]],
        model: str,
    ) -> str:
        with trace(f"CustomWithMemoryEval-{ub_id}"):
            context = EvaluationContext(
                workflow_state=workflow_state,
                eval_instructions=eval_instructions,
                criteria=criteria,
            )

            total_max_points = self._calculate_total_points(criteria)

            def agent_instructions(run_context: RunContextWrapper[EvaluationContext], _agent: Agent) -> str:
                ctx = run_context.context

                criteria_text = ""
                for i, crit in enumerate(ctx.criteria):
                    criteria_text += f"\n## Criterion {i + 1}"
                    if crit.get("criterion_name"):
                        criteria_text += f": {crit['criterion_name']}"
                    criteria_text += f"\nMax Points: {crit.get('max_points', 0)}\n"
                    if crit.get("summary_instructions"):
                        criteria_text += f"Summary: {crit['summary_instructions']}\n"
                    if crit.get("grading_instructions"):
                        criteria_text += f"Grading: {crit['grading_instructions']}\n"
                    criteria_text += "\n"

                conversation_text = ""
                for i, ans in enumerate(ctx.workflow_state.answers):
                    conversation_text += f"\n{'=' * 60}\n"
                    conversation_text += f"Exchange {i + 1}:\n"
                    conversation_text += f"{'=' * 60}\n\n"
                    conversation_text += f"**User:** {ans.get('user_message', 'N/A')}\n\n"
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
                name="CustomWithMemoryEvaluator",
                instructions=agent_instructions,
                model=model,
                model_settings=ModelSettings(temperature=0.3, max_tokens=2048),
            )

            result = await Runner.run(agent, "", context=context)
            evaluation_text = result.final_output_as(str)

            if isinstance(evaluation_text, str):
                evaluation_text = evaluation_text.strip()

            return evaluation_text
