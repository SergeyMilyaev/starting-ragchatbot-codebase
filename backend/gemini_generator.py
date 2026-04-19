from google import genai
from google.genai import types
from typing import List, Optional, Dict, Any

from base_generator import BaseAIGenerator


class GeminiAIGenerator(BaseAIGenerator):
    """Handles interactions with Google's Gemini API for generating responses."""

    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- **One search per query maximum**
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


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

        config_kwargs: Dict[str, Any] = {
            "system_instruction": system_content,
            "temperature": 0,
            "max_output_tokens": 800,
        }
        if tools:
            config_kwargs["tools"] = self._convert_tools(tools)

        response = self.client.models.generate_content(
            model=self.model,
            contents=query,
            config=types.GenerateContentConfig(**config_kwargs),
        )

        part = response.candidates[0].content.parts[0]
        if hasattr(part, "function_call") and part.function_call and tool_manager:
            return self._handle_tool_execution(
                response, query, system_content, config_kwargs.get("tools"), tool_manager
            )

        return response.text

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

    def _handle_tool_execution(
        self,
        initial_response,
        original_query: str,
        system_content: str,
        gemini_tools,
        tool_manager,
    ) -> str:
        user_turn = types.Content(role="user", parts=[types.Part(text=original_query)])
        model_turn = types.Content(
            role="model",
            parts=initial_response.candidates[0].content.parts,
        )

        result_parts = []
        for part in initial_response.candidates[0].content.parts:
            if hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                tool_result = tool_manager.execute_tool(fc.name, **dict(fc.args))
                result_parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=fc.name,
                            response={"result": tool_result},
                        )
                    )
                )

        tool_turn = types.Content(role="user", parts=result_parts)

        final_response = self.client.models.generate_content(
            model=self.model,
            contents=[user_turn, model_turn, tool_turn],
            config=types.GenerateContentConfig(
                system_instruction=system_content,
                temperature=0,
                max_output_tokens=800,
            ),
        )

        return final_response.text
