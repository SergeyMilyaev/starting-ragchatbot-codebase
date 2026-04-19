"""
Tests for GeminiAIGenerator — verifies tool conversion, API call structure,
tool execution flow, and response extraction for the Gemini provider path.
"""
import pytest
from unittest.mock import MagicMock, patch, call


# ── helpers ────────────────────────────────────────────────────────────────

SEARCH_TOOL_DEF = {
    "name": "search_course_content",
    "description": "Search course materials",
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}

OUTLINE_TOOL_DEF = {
    "name": "get_course_outline",
    "description": "Get the full outline of a course",
    "input_schema": {
        "type": "object",
        "properties": {"course_title": {"type": "string"}},
        "required": ["course_title"],
    },
}


def make_text_response(text):
    """Gemini response that resolves to plain text (no tool call)."""
    resp = MagicMock()
    resp.text = text
    part = MagicMock()
    part.function_call = None
    resp.candidates = [MagicMock()]
    resp.candidates[0].content.parts = [part]
    return resp


def make_tool_response(tool_name, tool_args):
    """Gemini response containing a single function_call part."""
    resp = MagicMock()
    fc = MagicMock()
    fc.name = tool_name
    fc.args = tool_args
    part = MagicMock()
    part.function_call = fc
    resp.candidates = [MagicMock()]
    resp.candidates[0].content.parts = [part]
    return resp


# ── fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def mock_genai():
    """Patch the google.genai module used inside gemini_generator."""
    with patch("gemini_generator.genai") as mock:
        yield mock


@pytest.fixture
def mock_types():
    """Patch google.genai.types used for Gemini object construction."""
    with patch("gemini_generator.types") as mock:
        # Make type constructors return identifiable objects
        mock.FunctionDeclaration.side_effect = lambda **kw: {"fd": kw}
        mock.Tool.side_effect = lambda function_declarations: {"tool": function_declarations}
        mock.GenerateContentConfig.side_effect = lambda **kw: {"cfg": kw}
        mock.Content.side_effect = lambda **kw: {"content": kw}
        mock.Part.side_effect = lambda **kw: {"part": kw}
        mock.FunctionResponse.side_effect = lambda **kw: {"fr": kw}
        yield mock


@pytest.fixture
def generator(mock_genai, mock_types):
    from gemini_generator import GeminiAIGenerator
    return GeminiAIGenerator(api_key="test_key", model="gemini-test-model")


# ── direct response ────────────────────────────────────────────────────────

def test_direct_response_returned_as_text(generator, mock_genai):
    mock_genai.Client.return_value.models.generate_content.return_value = (
        make_text_response("Direct Gemini answer")
    )
    result = generator.generate_response("What is MCP?")
    assert result == "Direct Gemini answer"


def test_generate_content_called_once_for_direct_response(generator, mock_genai):
    mock_genai.Client.return_value.models.generate_content.return_value = (
        make_text_response("answer")
    )
    generator.generate_response("query")
    assert mock_genai.Client.return_value.models.generate_content.call_count == 1


def test_model_name_passed_to_generate_content(generator, mock_genai):
    mock_genai.Client.return_value.models.generate_content.return_value = (
        make_text_response("answer")
    )
    generator.generate_response("query")
    call_kwargs = mock_genai.Client.return_value.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-test-model"


def test_query_passed_as_contents(generator, mock_genai):
    mock_genai.Client.return_value.models.generate_content.return_value = (
        make_text_response("answer")
    )
    generator.generate_response("What is Python?")
    call_kwargs = mock_genai.Client.return_value.models.generate_content.call_args.kwargs
    assert call_kwargs["contents"] == "What is Python?"


def test_system_prompt_in_config(generator, mock_genai, mock_types):
    mock_genai.Client.return_value.models.generate_content.return_value = (
        make_text_response("answer")
    )
    generator.generate_response("query")
    cfg_call = mock_types.GenerateContentConfig.call_args.kwargs
    assert "system_instruction" in cfg_call
    assert len(cfg_call["system_instruction"]) > 0


def test_conversation_history_appended_to_system(generator, mock_genai, mock_types):
    mock_genai.Client.return_value.models.generate_content.return_value = (
        make_text_response("answer")
    )
    generator.generate_response("query", conversation_history="User: Hi\nAssistant: Hello")
    cfg_call = mock_types.GenerateContentConfig.call_args.kwargs
    assert "Previous conversation:" in cfg_call["system_instruction"]
    assert "User: Hi" in cfg_call["system_instruction"]


# ── tool definition conversion ─────────────────────────────────────────────

def test_convert_tools_uses_input_schema_as_parameters(generator, mock_types):
    """Anthropic uses 'input_schema'; Gemini needs 'parameters' — _convert_tools must adapt."""
    mock_genai_client = MagicMock()
    generator.client = mock_genai_client

    generator._convert_tools([SEARCH_TOOL_DEF])

    mock_types.FunctionDeclaration.assert_called_once_with(
        name="search_course_content",
        description="Search course materials",
        parameters=SEARCH_TOOL_DEF["input_schema"],
    )


def test_convert_tools_wraps_declarations_in_tool_object(generator, mock_types):
    result = generator._convert_tools([SEARCH_TOOL_DEF])
    mock_types.Tool.assert_called_once()
    assert isinstance(result, list)
    assert len(result) == 1


def test_convert_tools_handles_multiple_tools(generator, mock_types):
    result = generator._convert_tools([SEARCH_TOOL_DEF, OUTLINE_TOOL_DEF])
    assert mock_types.FunctionDeclaration.call_count == 2


def test_tools_included_in_config_when_provided(generator, mock_genai, mock_types):
    mock_genai.Client.return_value.models.generate_content.return_value = (
        make_text_response("answer")
    )
    generator.generate_response("query", tools=[SEARCH_TOOL_DEF])
    cfg_call = mock_types.GenerateContentConfig.call_args.kwargs
    assert "tools" in cfg_call


def test_no_tools_in_config_when_not_provided(generator, mock_genai, mock_types):
    mock_genai.Client.return_value.models.generate_content.return_value = (
        make_text_response("answer")
    )
    generator.generate_response("query")
    cfg_call = mock_types.GenerateContentConfig.call_args.kwargs
    assert "tools" not in cfg_call


# ── tool execution flow ────────────────────────────────────────────────────

def test_function_call_triggers_tool_execution(generator, mock_genai, mock_types):
    mock_client = mock_genai.Client.return_value
    mock_client.models.generate_content.side_effect = [
        make_tool_response("search_course_content", {"query": "MCP"}),
        make_text_response("MCP is a protocol"),
    ]

    mock_manager = MagicMock()
    mock_manager.execute_tool.return_value = "Tool result: MCP content"

    result = generator.generate_response(
        "What is MCP?", tools=[SEARCH_TOOL_DEF], tool_manager=mock_manager
    )

    mock_manager.execute_tool.assert_called_once_with(
        "search_course_content", query="MCP"
    )
    assert result == "MCP is a protocol"
    assert mock_client.models.generate_content.call_count == 2


def test_tool_result_sent_as_function_response(generator, mock_genai, mock_types):
    """The tool output must be wrapped in types.FunctionResponse, not a plain string."""
    mock_client = mock_genai.Client.return_value
    mock_client.models.generate_content.side_effect = [
        make_tool_response("search_course_content", {"query": "MCP"}),
        make_text_response("answer"),
    ]
    mock_manager = MagicMock()
    mock_manager.execute_tool.return_value = "course content here"

    generator.generate_response("query", tools=[SEARCH_TOOL_DEF], tool_manager=mock_manager)

    mock_types.FunctionResponse.assert_called_once_with(
        name="search_course_content",
        response={"result": "course content here"},
    )


def test_second_generate_content_call_has_no_tools(generator, mock_genai, mock_types):
    """The follow-up call after tool execution must NOT include tools."""
    mock_client = mock_genai.Client.return_value
    mock_client.models.generate_content.side_effect = [
        make_tool_response("search_course_content", {"query": "MCP"}),
        make_text_response("answer"),
    ]
    mock_manager = MagicMock()
    mock_manager.execute_tool.return_value = "result"

    generator.generate_response("query", tools=[SEARCH_TOOL_DEF], tool_manager=mock_manager)

    # The second GenerateContentConfig call should not include tools
    second_cfg_call = mock_types.GenerateContentConfig.call_args_list[1].kwargs
    assert "tools" not in second_cfg_call, (
        "Follow-up Gemini call must not include tools to avoid infinite function-call loops"
    )


def test_second_call_includes_conversation_turns(generator, mock_genai, mock_types):
    """Follow-up call must include user + model + tool_result turns as contents."""
    mock_client = mock_genai.Client.return_value
    mock_client.models.generate_content.side_effect = [
        make_tool_response("search_course_content", {"query": "MCP"}),
        make_text_response("answer"),
    ]
    mock_manager = MagicMock()
    mock_manager.execute_tool.return_value = "result"

    generator.generate_response("What is MCP?", tools=[SEARCH_TOOL_DEF], tool_manager=mock_manager)

    second_call_kwargs = mock_client.models.generate_content.call_args_list[1].kwargs
    contents = second_call_kwargs["contents"]
    # Should be a list: [user_turn, model_turn, tool_turn]
    assert isinstance(contents, list)
    assert len(contents) == 3


def test_no_tool_execution_when_no_function_call(generator, mock_genai):
    """When the response has no function_call, tool_manager must not be called."""
    mock_genai.Client.return_value.models.generate_content.return_value = (
        make_text_response("Direct answer")
    )
    mock_manager = MagicMock()
    result = generator.generate_response(
        "What is Python?", tools=[SEARCH_TOOL_DEF], tool_manager=mock_manager
    )
    mock_manager.execute_tool.assert_not_called()
    assert result == "Direct answer"


def test_multiple_function_calls_all_executed(generator, mock_genai, mock_types):
    """All function_call parts in a response must be executed."""
    fc_a = MagicMock()
    fc_a.name = "search_course_content"
    fc_a.args = {"query": "A"}

    fc_b = MagicMock()
    fc_b.name = "get_course_outline"
    fc_b.args = {"course_title": "MCP"}

    part_a = MagicMock()
    part_a.function_call = fc_a
    part_b = MagicMock()
    part_b.function_call = fc_b

    resp = MagicMock()
    resp.candidates = [MagicMock()]
    resp.candidates[0].content.parts = [part_a, part_b]

    mock_client = mock_genai.Client.return_value
    mock_client.models.generate_content.side_effect = [resp, make_text_response("answer")]

    mock_manager = MagicMock()
    mock_manager.execute_tool.return_value = "result"

    generator.generate_response("query", tools=[SEARCH_TOOL_DEF, OUTLINE_TOOL_DEF], tool_manager=mock_manager)

    assert mock_manager.execute_tool.call_count == 2


# ── system prompt completeness ─────────────────────────────────────────────

def test_gemini_system_prompt_includes_outline_guidance():
    """GeminiAIGenerator.SYSTEM_PROMPT must include the outline tool guidance
    (same as AIGenerator) so course-outline queries work consistently."""
    from gemini_generator import GeminiAIGenerator
    assert "get_course_outline" in GeminiAIGenerator.SYSTEM_PROMPT, (
        "GeminiAIGenerator.SYSTEM_PROMPT is missing Course Outline Queries section"
    )


def test_anthropic_and_gemini_prompts_both_mention_outline():
    """Both generators must have the outline guidance so behaviour is consistent."""
    from ai_generator import AIGenerator
    from gemini_generator import GeminiAIGenerator
    assert "get_course_outline" in AIGenerator.SYSTEM_PROMPT
    assert "get_course_outline" in GeminiAIGenerator.SYSTEM_PROMPT
