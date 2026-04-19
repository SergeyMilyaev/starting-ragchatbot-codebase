from google import genai
from google.genai import types
from typing import List, Optional, Dict, Any

from base_generator import BaseAIGenerator


class GeminiAIGenerator(BaseAIGenerator):
    """Handles interactions with Google's Gemini API for generating responses."""

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
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        gemini_tools = self._convert_tools(tools) if tools else None
        base_config_kwargs: Dict[str, Any] = {
            "system_instruction": system_content,
            "temperature": 0,
            "max_output_tokens": 800,
        }
        if gemini_tools:
            base_config_kwargs["tools"] = gemini_tools

        contents: List = [types.Content(role="user", parts=[types.Part(text=query)])]

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(**base_config_kwargs),
        )

        for _ in range(2):
            parts = response.candidates[0].content.parts
            has_function_call = any(
                hasattr(p, "function_call") and p.function_call for p in parts
            )
            if not has_function_call or not tool_manager:
                return response.text

            contents.append(types.Content(role="model", parts=parts))

            result_parts = []
            for part in parts:
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    try:
                        tool_result = tool_manager.execute_tool(fc.name, **dict(fc.args))
                    except Exception as e:
                        tool_result = f"Tool execution failed: {e}"
                    result_parts.append(
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=fc.name,
                                response={"result": tool_result},
                            )
                        )
                    )

            contents.append(types.Content(role="user", parts=result_parts))

            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(**base_config_kwargs),
            )

        # Check if last follow-up response is already text (no more tool calls needed)
        parts = response.candidates[0].content.parts
        if not any(hasattr(p, "function_call") and p.function_call for p in parts):
            return response.text

        # Rounds exhausted but model still wants tools — force final synthesis without tools
        final_config = {k: v for k, v in base_config_kwargs.items() if k != "tools"}
        final_response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(**final_config),
        )
        return final_response.text

    def _convert_tools(self, anthropic_tools: List[Dict]) -> List[types.Tool]:
        """Convert Anthropic-format tool dicts to Gemini Tool objects.

        Anthropic uses "input_schema" key; Gemini uses "parameters" kwarg —
        the inner JSON Schema content is otherwise identical.
        """
        declarations = []
        for tool in anthropic_tools:
            declarations.append(
                types.FunctionDeclaration(
                    name=tool["name"],
                    description=tool["description"],
                    parameters=tool["input_schema"],
                )
            )
        return [types.Tool(function_declarations=declarations)]
