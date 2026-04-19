import anthropic
from typing import List, Optional, Dict, Any

from base_generator import BaseAIGenerator

class AIGenerator(BaseAIGenerator):
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- You may perform up to **two sequential tool calls** when the first result reveals additional information is needed
- Chain tools only when necessary (e.g. get course outline to identify a lesson, then search for related content)
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


Course Outline Queries:
- For questions about course structure, outline, or lesson list: use the get_course_outline tool
- Present the result with: course title as a heading, the course link, then a numbered list of lessons (lesson number and title)

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        messages: List[Dict[str, Any]] = [{"role": "user", "content": query}]

        call_params: Dict[str, Any] = {**self.base_params, "system": system_content}
        if tools:
            call_params["tools"] = tools
            call_params["tool_choice"] = {"type": "auto"}

        response = self.client.messages.create(**call_params, messages=messages)

        for _ in range(2):
            if response.stop_reason != "tool_use" or not tool_manager:
                return response.content[0].text

            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    try:
                        result = tool_manager.execute_tool(block.name, **block.input)
                    except Exception as e:
                        result = f"Tool execution failed: {e}"
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            messages.append({"role": "user", "content": tool_results})
            response = self.client.messages.create(**call_params, messages=messages)

        # Check if last follow-up response is already text (no more tool calls needed)
        if response.stop_reason != "tool_use":
            return response.content[0].text

        # Rounds exhausted but model still wants tools — force final synthesis without tools
        final_params = {**self.base_params, "system": system_content, "messages": messages}
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text