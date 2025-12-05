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
            
            topic = ""
            if ctx.state.answers:
                first_answer = ctx.state.answers[0]
                topic = first_answer.get('topic', '')
            
            conversation_history = ""
            if len(ctx.state.answers) > 1:
                conversation_history = "\n# Recent conversation:\n"
                for ans in ctx.state.answers[-3:]:
                    if ans.get('user_message'):
                        conversation_history += f"Student: {ans['user_message']}\n"
                    if ans.get('tutor_response'):
                        conversation_history += f"You: {ans['tutor_response']}\n"
            
            last_answer = ctx.state.answers[-1] if ctx.state.answers else {}
            
            if last_answer.get('waiting_for_answer'):
                student_message = last_answer.get('user_message', '')
                
                question_indicators = [
                    '?', 'what', 'how', 'why', 'could you', 'can you', 'explain', 'help', 
                    'don\'t understand', 'unclear', 'confused', 'mean', 'clarify',
                    'не розумію', 'поясни', 'що', 'як', 'чому', 'допоможи', 'розкажи',
                    'не зрозумів', 'не зрозуміла', 'шо', 'допоможіть', 'підкажи'
                ]
                is_question = any(indicator in student_message.lower() for indicator in question_indicators)
                
                assignment_length = len(last_answer.get('assignment', '').split())
                answer_length = len(student_message.split())
                seems_incomplete = answer_length < assignment_length * 0.3
                
                if is_question or seems_incomplete:
                    assignment_text = last_answer.get('assignment', '')
                    return f"""You are a warm, helpful English tutor. The student needs help with the assignment.

{conversation_history}

# Current assignment:
{assignment_text}

# Topic: {topic}

# Student's message:
{student_message}

# Your task:
The student is asking for help or clarification. Be supportive and conversational!

You can:
- Break down the task step by step
- Give examples similar to what they need to do
- Explain vocabulary or grammar in simple terms
- Provide hints without giving the exact answer
- Use a friendly, encouraging tone
- If they asked in Ukrainian, you can acknowledge it and respond in English in a warm way

IMPORTANT: Be natural and conversational, not robotic. Show empathy and make learning feel easy.

Keep your response helpful and friendly (max 250 words).

Remind them gently that they can try when they feel ready."""
                
                else:
                    return f"""The student sent: "{student_message}"

This doesn't look like a complete answer to the assignment. 

Respond warmly and naturally:
- Acknowledge what they sent
- Gently ask them to provide the full answer with all tasks completed
- Offer help if they're stuck
- Keep it conversational and friendly

Max 120 words."""
            
            if last_answer and last_answer.get('graded'):
                evaluation = last_answer.get('evaluation', {})
                student_answer = last_answer.get('answer', '')
                
                feedback_parts = []
                
                if evaluation.get('correct'):
                    positive_responses = [
                        "✅ Excellent work! That's exactly right.",
                        "✅ Perfect! You nailed it.",
                        "✅ Great job! All correct.",
                        "✅ Well done! Your answers are spot on."
                    ]
                    import random
                    feedback_parts.append(random.choice(positive_responses))
                else:
                    feedback_parts.append("Let me check your answers:\n")
                    for error in evaluation.get('errors', []):
                        feedback_parts.append(f"❌ {error}")
                    feedback_parts.append(f"\n{evaluation.get('feedback', '')}")
                    feedback_parts.append("\nNo worries though – let's keep practicing! 💪")
                
                feedback_parts.append(f"\n\n**Ready for the next challenge? Assignment #{current_assignment_index + 1}**\n")
                
                return f"""You are a friendly, encouraging English tutor. Give feedback naturally and present the next assignment.

{conversation_history}

# Student's previous answer:
{student_answer}

# Your feedback:
{chr(10).join(feedback_parts)}

# Now create the NEXT assignment:
- Topic: {topic}
- Learning Goal: {learning_goal}
- Format reference: {examples}

CRITICAL RULES:
1. Present ONLY the clean assignment with numbered tasks
2. NO meta-sections like "Learning Goal" or "Assignment Format"
3. NO internal instructions visible to student
4. Add a brief, friendly intro (1-2 sentences) that feels natural
5. Make it relevant to: {topic}
6. Be conversational, not formal

Generate assignment #{current_assignment_index + 1} now."""
            
            elif last_answer and not last_answer.get('graded'):
                return "Wait for the student to provide their full answer before proceeding."
            
            else:
                if not topic:
                    return f"""You are a friendly English tutor starting a conversation.

{conversation_history}

# Your task:
Ask the student what topic they'd like to practice with today.

# Guidance:
{flexible_part}

Be warm and conversational. Give 2-3 interesting example topics.

Keep it natural and inviting (max 100 words)."""
                
                else:
                    return f"""You are a friendly English tutor presenting the first assignment.

{conversation_history}

# Topic chosen: {topic}

# Assignment instructions:
- Learning Goal: {learning_goal}
- Format reference: {examples}

CRITICAL RULES:
1. Present ONLY the clean assignment with numbered tasks
2. NO meta-sections like "Learning Goal" or "Assignment Format"
3. NO internal instructions visible to student
4. Add a brief, warm intro (1-2 sentences) before the tasks
5. Make it about: {topic}
6. Be conversational and encouraging

Generate assignment #1 now."""
        
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
- Be lenient with minor mistakes if meaning is clear

Return JSON:
{{
  "correct": true/false,
  "errors": ["error explanation", ...],
  "feedback": "encouraging overall feedback"
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
            
            if len(state.answers) == 0:
                state.answers.append({
                    "assignment_index": 0,
                    "topic": "",
                    "assignment": "",
                    "answer": "",
                    "timestamp": datetime.now().isoformat(),
                    "graded": False,
                    "waiting_for_topic": True,
                    "user_message": "",
                    "tutor_response": ""
                })
                
                tutor = self.create_tutor_agent(context, specs, template.get("model", "gpt-4o"))
                result = await Runner.run(tutor, "", context=context)
                response = result.final_output_as(str)
                
                state.answers[0]['assignment'] = response
                await xano.save_workflow_state(state)
                
                return response
            
            last_answer = state.answers[-1] if state.answers else {}
            
            if last_answer.get('waiting_for_topic'):
                topic = user_message.strip()
                
                if len(topic.split()) <= 2 or '?' in topic:
                    last_answer['user_message'] = user_message
                    
                    tutor = self.create_tutor_agent(context, specs, template.get("model", "gpt-4o"))
                    result = await Runner.run(tutor, user_message, context=context)
                    response = result.final_output_as(str)
                    
                    last_answer['tutor_response'] = response
                    await xano.save_workflow_state(state)
                    
                    return response
                
                last_answer['topic'] = topic
                last_answer['waiting_for_topic'] = False
                last_answer['waiting_for_answer'] = True
                
                tutor = self.create_tutor_agent(context, specs, template.get("model", "gpt-4o"))
                result = await Runner.run(tutor, "", context=context)
                response = result.final_output_as(str)
                
                last_answer['assignment'] = response
                await xano.save_workflow_state(state)
                
                return response
            
            if last_answer and last_answer.get('waiting_for_answer'):
                student_message = user_message.strip()
                
                question_indicators = [
                    '?', 'what', 'how', 'why', 'could you', 'can you', 'explain', 'help', 
                    'don\'t understand', 'unclear', 'confused', 'mean', 'clarify',
                    'не розумію', 'поясни', 'що', 'як', 'чому', 'допоможи', 'розкажи',
                    'не зрозумів', 'не зрозуміла', 'шо', 'допоможіть', 'підкажи'
                ]
                is_question = any(indicator in student_message.lower() for indicator in question_indicators)
                
                short_response_indicators = ['ok', 'okay', 'thanks', 'got it', 'understand', 'yes', 'no', 'wait']
                is_short_response = len(student_message.split()) <= 3 and any(indicator in student_message.lower() for indicator in short_response_indicators)
                
                assignment_length = len(last_answer.get('assignment', '').split())
                answer_length = len(student_message.split())
                seems_incomplete = answer_length < assignment_length * 0.3
                
                if is_question or is_short_response or seems_incomplete:
                    last_answer['user_message'] = user_message
                    
                    tutor = self.create_tutor_agent(context, specs, template.get("model", "gpt-4o"))
                    result = await Runner.run(tutor, user_message, context=context)
                    response = result.final_output_as(str)
                    
                    last_answer['tutor_response'] = response
                    await xano.save_workflow_state(state)
                    
                    return response
                
                last_answer['answer'] = user_message
                last_answer['timestamp'] = datetime.now().isoformat()
                last_answer['waiting_for_answer'] = False
                
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
                
                topic = state.answers[0].get('topic', '')
                
                state.answers.append({
                    "assignment_index": state.current_question_index,
                    "topic": topic,
                    "assignment": response,
                    "answer": "",
                    "timestamp": datetime.now().isoformat(),
                    "graded": False,
                    "waiting_for_answer": True,
                    "user_message": "",
                    "tutor_response": ""
                })
                await xano.save_workflow_state(state)
                
                return response
            
            else:
                tutor = self.create_tutor_agent(context, specs, template.get("model", "gpt-4o"))
                result = await Runner.run(tutor, "", context=context)
                response = result.final_output_as(str)
                
                topic = state.answers[0].get('topic', '') if state.answers else ''
                
                state.answers.append({
                    "assignment_index": state.current_question_index,
                    "topic": topic,
                    "assignment": response,
                    "answer": "",
                    "timestamp": datetime.now().isoformat(),
                    "graded": False,
                    "waiting_for_answer": True,
                    "user_message": "",
                    "tutor_response": ""
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
            
            total_max_points = self._calculate_total_points(criteria)
            
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

For each criterion:
1. Review the assignments
2. Assess how well the student met the criterion
3. Assign a grade (0 to max_points for that criterion)
4. Provide clear reasoning with specific examples

Format your response as:

# Evaluation Report

## Criterion 1: [Name]
**Assessment:** [Detailed assessment with examples]
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
                name="AnalogousFullEvaluator",
                instructions=agent_instructions,
                model=model,
                model_settings=ModelSettings(temperature=0.3, max_tokens=2048)
            )
            
            result = await Runner.run(agent, "", context=context)
            evaluation_text = result.final_output_as(str)
            
            if isinstance(evaluation_text, str):
                evaluation_text = evaluation_text.strip()
            
            return evaluation_text