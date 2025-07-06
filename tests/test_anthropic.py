"""Test the Anthropic Wrapper."""
import asyncio
from copy import deepcopy
import os
from faker import Faker
import pytest
import random
from dotenv import load_dotenv
from sik_llms import (
    Client,
    system_message,
    user_message,
    assistant_message,
    Tool,
    RegisteredClients,
    Anthropic,
    AnthropicTools,
    TextChunkEvent,
    TextResponse,
    ReasoningEffort,
    ThinkingChunkEvent,
)
from sik_llms.anthropic import SUPPORTED_ANTHROPIC_MODELS, _convert_messages
from sik_llms.models_base import ImageContent, ImageSourceType
from tests.conftest import ANTHROPIC_TEST_MODEL, ANTRHOPIC_TEST_THINKING_MODEL
load_dotenv()


@pytest.mark.asyncio
@pytest.mark.skipif(os.getenv('ANTHROPIC_API_KEY') is None, reason="ANTHROPIC_API_KEY is not set")
class TestAnthropic:
    """Test the Anthropic Completion Wrapper."""

    async def test__all_models(self):
        """Test all models concurrently for better performance."""
        async def test_single_model(model_name: str) -> TextResponse:
            """Test a single model and return results."""
            client = Anthropic(model_name=model_name)
            messages = [
                system_message("You are a helpful assistant."),
                user_message("Respond with exactly \"42\"."),
            ]
            response = await client.run_async(messages=messages)
            assert isinstance(response, TextResponse)
            assert '42' in response.response
            assert response.input_tokens > 0
            assert response.output_tokens > 0
            assert response.total_tokens > 0
            assert response.input_cost > 0
            assert response.output_cost > 0
            assert response.total_cost > 0
            assert response.duration_seconds > 0
            return response

        # Run all models concurrently
        model_names = list(SUPPORTED_ANTHROPIC_MODELS.keys())
        results = await asyncio.gather(
            *(test_single_model(model) for model in model_names),
            return_exceptions=True,
        )
        assert len(results) == len(model_names)
        assert all(isinstance(result, TextResponse) for result in results)


class TestAnthropicSync:  # noqa: D101

    def test__Anthropic_registration(self):
        assert Client.is_registered(RegisteredClients.ANTHROPIC)

    def test__Anthropic__instantiate(self):
        model = Client.instantiate(
            client_type=RegisteredClients.ANTHROPIC,
            model_name=ANTHROPIC_TEST_MODEL,
            api_key='fake_key',
        )
        assert isinstance(model, Anthropic)
        assert model.model == ANTHROPIC_TEST_MODEL
        assert model.client is not None
        assert model.client.api_key == 'fake_key'

    def test__Anthropic_instantiate___parameters(self):
        model = Client.instantiate(
            client_type=RegisteredClients.ANTHROPIC,
            model_name=ANTHROPIC_TEST_MODEL,
            api_key='fake_key',
            temperature=0.5,
            max_tokens=100,
        )
        assert isinstance(model, Anthropic)
        assert model.model == ANTHROPIC_TEST_MODEL
        assert model.model_parameters['temperature'] == 0.5
        assert model.model_parameters['max_tokens'] == 100

    def test__Anthropic_instantiate___default_max_tokens(self):
        model = Client.instantiate(
            client_type=RegisteredClients.ANTHROPIC,
            model_name=ANTHROPIC_TEST_MODEL,
            api_key='fake_key',
        )
        assert isinstance(model, Anthropic)
        assert model.model == ANTHROPIC_TEST_MODEL
        assert model.model_parameters['max_tokens'] > 0


@pytest.mark.asyncio
@pytest.mark.skipif(os.getenv('ANTHROPIC_API_KEY') is None, reason="ANTHROPIC_API_KEY is not set")
class TestAnthropicAsync:  # noqa: D101

    async def test__async_anthropic_completion_wrapper_call(self):
        """Test the Anthropic wrapper with multiple concurrent calls."""
        # Create an instance of the wrapper
        client = Anthropic(
            model_name=ANTHROPIC_TEST_MODEL,
            max_tokens=100,
        )
        messages = [
            system_message("You are a helpful assistant."),
            user_message("What is the capital of France?"),
        ]

        async def run_model():  # noqa: ANN202
            chunks = []
            summary = None
            try:
                async for response in client.stream(messages=messages):
                    if isinstance(response, TextChunkEvent):
                        chunks.append(response)
                    elif isinstance(response, TextResponse):
                        summary = response
                return chunks, summary
            except Exception:
                return [], None

        results = await asyncio.gather(*(run_model() for _ in range(10)))
        passed_tests = []

        for chunks, summary in results:
            response = ''.join([chunk.content for chunk in chunks])
            passed_tests.append(
                'Paris' in response
                and isinstance(summary, TextResponse)
                and summary.input_tokens > 0
                and summary.output_tokens > 0
                and summary.input_cost > 0
                and summary.output_cost > 0
                and summary.duration_seconds > 0,
            )

        assert sum(passed_tests) / len(passed_tests) >= 0.9, (
            f"Only {sum(passed_tests)} out of {len(passed_tests)} tests passed."
        )

    async def test__Anthropic__instantiate__stream(self):
        model = Client.instantiate(
            client_type=RegisteredClients.ANTHROPIC,
            model_name=ANTHROPIC_TEST_MODEL,
        )
        responses = []
        async for response in model.stream(messages=[user_message("What is the capital of France?")]):  # noqa: E501
            if isinstance(response, TextChunkEvent):
                responses.append(response)

        assert len(responses) > 0
        assert 'Paris' in ''.join([response.content for response in responses])


class TestConvertMessages:
    """Test the _convert_messages method."""

    def test__Anthropic___convert_messages__empty(self):
        system_messages, other_messages = _convert_messages([])
        assert not system_messages
        assert not other_messages

    def test__Anthropic___convert_messages__no_system(self):
        message_user = "What is the capital of France?"
        messages = [
            user_message(message_user),
        ]
        original_messages = deepcopy(messages)
        converted_system, converted_other = _convert_messages(messages)
        assert messages == original_messages  # test no side effects
        assert not converted_system
        assert len(converted_other) == 1
        assert converted_other == [{
            'role': 'user',
            'content': message_user,
        }]

    def test__Anthropic___convert_messages__system__user(self):
        message_system = "You are a helpful assistant."
        message_user = "What is the capital of France?"
        messages = [
            system_message(message_system),
            user_message(message_user),
        ]
        original_messages = deepcopy(messages)
        converted_system, converted_other = _convert_messages(messages)
        assert messages == original_messages  # test no side effects
        assert len(converted_system) == 1
        assert len(converted_other) == 1
        assert converted_system == [{
            'type': 'text',
            'text': message_system,
        }]
        assert converted_other == [{
            'role': 'user',
            'content': message_user,
        }]

    def test__Anthropic___convert_messages__system__user__assistant(self):
        message_system = "You are a helpful assistant."
        message_user = "What is the capital of France?"
        message_assistant = "The capital of France is Paris."

        messages = [
            system_message(message_system),
            user_message(message_user),
            assistant_message(message_assistant),
        ]
        original_messages = deepcopy(messages)
        converted_system, converted_other = _convert_messages(messages)
        assert messages == original_messages  # test no side effects
        assert len(converted_system) == 1
        assert len(converted_other) == 2
        assert converted_system == [{
            'type': 'text',
            'text': message_system,
        }]
        assert converted_other == [
            {
                'role': 'user',
                'content': message_user,
            },
            {
                'role': 'assistant',
                'content': message_assistant,
            },
        ]

    def test__Anthropic___convert_messages__multiple_system_with_cache_control__user(self):
        message_system = "You are a helpful assistant."
        message_cache = "This is some cached information."
        message_user = "What is the capital of France?"
        messages = [
            system_message(message_system),
            system_message(message_cache, cache_control={'type': 'ephemeral'}),
            user_message(message_user),
        ]
        original_messages = deepcopy(messages)
        converted_system, converted_other = _convert_messages(messages)
        assert messages == original_messages  # test no side effects
        assert len(converted_system) == 2
        assert len(converted_other) == 1
        assert converted_system == [
            {
                'type': 'text',
                'text': message_system,
            },
            {
                'type': 'text',
                'text': message_cache,
                'cache_control': {'type': 'ephemeral'},
            },
        ]
        assert converted_other == [{
            'role': 'user',
            'content': message_user,
        }]

    def test__Anthropic___convert_messages__no_system_with_cache_control__user__assistant(self):
        # based on example from : https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
        # system=[
        # {
        #     "type": "text",
        #     "text": "You are an AI assistant tasked with analyzing literary works. Your goal is to provide insightful commentary on themes, characters, and writing style.\n",  # noqa: E501
        # },
        # {
        #     "type": "text",
        #     "text": "<the entire contents of 'Pride and Prejudice'>",
        #     "cache_control": {"type": "ephemeral"}
        # }
        # ],
        # messages=[{"role": "user", "content": "Analyze the major themes in 'Pride and Prejudice'."}],  # noqa: E501
        message_cache = "<the entire contents of 'Pride and Prejudice'>"
        message_user = "Analyze the major themes in 'Pride and Prejudice'."
        messages = [user_message(message_user)]
        original_messages = deepcopy(messages)

        # with list message_cache
        converted_system, converted_other = _convert_messages(
            messages, cache_content=[message_cache],
        )
        assert messages == original_messages  # test no side effects
        assert len(converted_system) == 1
        assert len(converted_other) == 1
        assert converted_system == [{
            'type': 'text',
            'text': message_cache,
            'cache_control': {'type': 'ephemeral'},
        }]
        assert converted_other == [{
            'role': 'user',
            'content': message_user,
        }]

        # with string message_cache
        converted_system, converted_other = _convert_messages(
            messages, cache_content=message_cache,
        )
        assert messages == original_messages  # test no side effects
        assert len(converted_system) == 1
        assert len(converted_other) == 1
        assert converted_system == [{
            'type': 'text',
            'text': message_cache,
            'cache_control': {'type': 'ephemeral'},
        }]
        assert converted_other == [{
            'role': 'user',
            'content': message_user,
        }]

    def test__Anthropic___convert_messages__multiple_system_with_cache_control__user__assistant(self):  # noqa: E501
        # based on example from : https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
        # system=[
        # {
        #     "type": "text",
        #     "text": "You are an AI assistant tasked with analyzing literary works. Your goal is to provide insightful commentary on themes, characters, and writing style.\n",  # noqa: E501
        # },
        # {
        #     "type": "text",
        #     "text": "<the entire contents of 'Pride and Prejudice'>",
        #     "cache_control": {"type": "ephemeral"}
        # }
        # ],
        # messages=[{"role": "user", "content": "Analyze the major themes in 'Pride and Prejudice'."}],  # noqa: E501
        message_system = "You are an AI assistant tasked with analyzing literary works."
        message_cache_1 = "<the entire contents of 'Pride and Prejudice'>"
        message_cache_2 = "<the entire contents of 'Treasure Island'>"
        message_user = "Analyze the major themes in 'Pride and Prejudice'."
        messages = [
            system_message(message_system),
            user_message(message_user),
        ]
        original_messages = deepcopy(messages)
        converted_system, converted_other = _convert_messages(
            messages, cache_content=[message_cache_1, message_cache_2],
        )
        assert messages == original_messages  # test no side effects
        assert len(converted_system) == 3
        assert len(converted_other) == 1
        assert converted_system == [
            {
                'type': 'text',
                'text': message_system,
            },
            {
                'type': 'text',
                'text': message_cache_1,
                'cache_control': {'type': 'ephemeral'},
            },
            {
                'type': 'text',
                'text': message_cache_2,
                'cache_control': {'type': 'ephemeral'},
            },
        ]
        assert converted_other == [{
            'role': 'user',
            'content': message_user,
        }]


    def test_complete_conversation(self):
        """Test conversion of a complete conversation with system, user, and assistant messages."""
        message_system = "You are a helpful assistant."
        message_user = "What is the capital of France?"
        message_assistant = "The capital of France is Paris."

        messages = [
            system_message(message_system),
            user_message(message_user),
            assistant_message(message_assistant),
        ]
        original_messages = deepcopy(messages)
        converted_system, converted_other = _convert_messages(messages)
        assert messages == original_messages  # test no side effects
        assert len(converted_system) == 1
        assert len(converted_other) == 2
        assert converted_system == [{
            'type': 'text',
            'text': message_system,
        }]
        assert converted_other == [
            {'role': 'user', 'content': message_user},
            {'role': 'assistant', 'content': message_assistant},
        ]

    def test_image_url_conversion(self):
        """Test conversion of messages containing URL images."""
        image = ImageContent.from_url("https://example.com/image.jpg")
        messages = [
            user_message([
                "What's in this image?",
                image,
            ]),
        ]

        converted_system, converted_other = _convert_messages(messages)
        assert not converted_system
        assert converted_other == [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": "https://example.com/image.jpg",
                    },
                },
            ],
        }]

    def test_image_base64_conversion(self):
        """Test conversion of messages containing base64 images."""
        image = ImageContent(
            source_type=ImageSourceType.BASE64,
            data="base64data",
            media_type="image/jpeg",
        )
        messages = [
            user_message([
                "What's in this image?",
                image,
            ]),
        ]

        converted_system, converted_other = _convert_messages(messages)
        assert not converted_system
        assert converted_other == [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": "base64data",
                    },
                },
            ],
        }]

    def test_cache_content_basic(self):
        """Test basic cache content insertion."""
        message_cache = "<cached content>"
        messages = [user_message("Analyze this.")]

        converted_system, converted_other = _convert_messages(
            messages=messages,
            cache_content=message_cache,
        )

        assert len(converted_system) == 1
        assert converted_system == [{
            'type': 'text',
            'text': message_cache,
            'cache_control': {'type': 'ephemeral'},
        }]
        assert converted_other == [{'role': 'user', 'content': "Analyze this."}]

    def test_cache_content_multiple(self):
        """Test multiple cache content items."""
        message_system = "You are an assistant."
        message_cache_1 = "<cache 1>"
        message_cache_2 = "<cache 2>"
        messages = [
            system_message(message_system),
            user_message("Analyze this."),
        ]

        converted_system, _ = _convert_messages(
            messages=messages,
            cache_content=[message_cache_1, message_cache_2],
        )

        assert len(converted_system) == 3
        assert converted_system == [
            {'type': 'text', 'text': message_system},
            {
                'type': 'text',
                'text': message_cache_1,
                'cache_control': {'type': 'ephemeral'},
            },
            {
                'type': 'text',
                'text': message_cache_2,
                'cache_control': {'type': 'ephemeral'},
            },
        ]

    def test_combined_features(self):
        """Test combining images, cache, and regular messages."""
        image1 = ImageContent.from_url("https://example.com/image1.jpg")
        image2 = ImageContent.from_url("https://example.com/image2.jpg")
        cache_content = ["Context 1", "Context 2"]

        messages = [
            system_message("You are an image analyst."),
            user_message(["First image:", image1]),
            assistant_message("I see a landscape."),
            user_message(["Second image:", image2, "Compare them."]),
        ]

        converted_system, converted_other = _convert_messages(
            messages=messages,
            cache_content=cache_content,
        )

        assert len(converted_system) == 3  # system + 2 cache messages
        assert len(converted_other) == 3  # 2 user messages + 1 assistant message
        assert all(msg.get('cache_control') == {'type': 'ephemeral'}
                  for msg in converted_system[1:])
        assert isinstance(converted_other[0]['content'], list)  # first image message
        assert isinstance(converted_other[2]['content'], list)  # second image message
        assert converted_other[1]['content'] == "I see a landscape."  # assistant message

    def test_content_stripping(self):
        """Test that text content is properly stripped in all contexts."""
        image = ImageContent.from_url("https://example.com/image.jpg")
        messages = [
            system_message("  System message with spaces  "),
            user_message([
                "  Text with spaces  ",
                image,
                "\nText with newlines\n",
            ]),
        ]
        cache_content = "  Cached content with spaces  "

        converted_system, converted_other = _convert_messages(
            messages=messages,
            cache_content=cache_content,
        )

        assert converted_system[0]['text'] == "System message with spaces"
        assert converted_system[1]['text'] == "Cached content with spaces"
        assert converted_other[0]['content'][0]['text'] == "Text with spaces"
        assert converted_other[0]['content'][2]['text'] == "Text with newlines"

@pytest.mark.skip_ci  # caching sporadically fails; let's skip this in CI
@pytest.mark.integration  # these tests make API calls
@pytest.mark.skipif(os.getenv('ANTHROPIC_API_KEY') is None, reason="ANTHROPIC_API_KEY is not set")
class TestAnthropicCaching:
    """
    Test the caching functionality of the Anthropic Wrapper.

    https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#why-am-i-seeing-the-error-attributeerror-beta-object-has-no-attribute-prompt-caching-in-python
    https://github.com/anthropics/anthropic-cookbook/blob/main/misc/prompt_caching.ipynb
    """

    @pytest.mark.parametrize('use_init', [True, False])
    def test__Anthropic__cache_control__expect_cache_miss_then_hit(self, use_init: bool):
        """
        Tests that the first call results in a cache-miss and the second call results in a
        cache-hit. The tokens written to the cache should be read on the second call.

        If use_init than we pass the cache content in the constructor, otherwise we create the
        system message with the cache content directly.
        """
        length = random.randint(15_000, 20_000)
        first_word = Faker().words(nb=1)[0]
        cache_content = f"{first_word} is the first word. {Faker().text(max_nb_chars=length)}"

        client = Anthropic(
            model_name=ANTHROPIC_TEST_MODEL,
            temperature=0.1,
            cache_content=[cache_content] if use_init else None,
        )

        if use_init:
            messages = [
                system_message("You are a helpful assistant."),
                user_message("What is the first word of this text? Only output the first word."),
            ]
        else:
            messages = [
                system_message("You are a helpful assistant."),
                system_message(
                    cache_content,
                    cache_control={'type': 'ephemeral'},
                ),
                user_message("What is the first word of this text? Only output the first word."),
            ]

        # first run should result in a cache-miss & write
        response = client(messages=messages)
        assert first_word in response.response
        assert response.input_tokens > 0
        assert response.output_tokens > 0
        assert response.cache_write_tokens > 0
        previous_cache_write_tokens = response.cache_write_tokens
        assert response.cache_read_tokens == 0
        expected_total_tokens = (
            response.input_tokens
            + response.output_tokens
            + response.cache_write_tokens
        )
        assert response.total_tokens == expected_total_tokens
        assert response.input_cost > 0
        assert response.output_cost > 0
        assert response.cache_write_cost > 0
        assert response.cache_read_cost == 0
        expected_total_cost = (
            response.input_cost
            + response.output_cost
            + response.cache_write_cost
        )
        assert response.total_cost == expected_total_cost

        # second run should result in a cache-hit
        response = client(messages=messages)
        assert first_word in response.response
        assert response.input_tokens > 0
        assert response.output_tokens > 0
        assert response.cache_write_tokens == 0
        assert response.cache_read_tokens == previous_cache_write_tokens
        expected_total_tokens = (
            response.input_tokens
            + response.output_tokens
            + response.cache_read_tokens
        )
        assert response.total_tokens == expected_total_tokens
        assert response.input_cost > 0
        assert response.output_cost > 0
        assert response.cache_write_cost == 0
        assert response.cache_read_cost > 0
        expected_total_cost = (
            response.input_cost
            + response.output_cost
            + response.cache_read_cost
        )
        assert response.total_cost == expected_total_cost

    def test__Anthropic__caching__with_tools(self, complex_weather_tool: Tool, restaurant_tool: Tool):  # noqa: E501

        # we need to make the text long enough to trigger the cache
        text = "[THE FOLLOWING IS TEST DATA FOR TESTING PURPOSES ONLY; **INGNORE THE FOLLOWING TEXT**]"  # noqa: E501
        text += Faker().text(max_nb_chars=15_000)
        complex_weather_tool.parameters[-1].description = "\n\n" + text

        client = AnthropicTools(
            model_name=ANTHROPIC_TEST_MODEL,
            tools=[complex_weather_tool, restaurant_tool],
            cache_tools=True,
        )
        messages=[
            user_message("What's the weather like in Tokyo in Celsius with forecast?"),
        ]
        response = client(messages=messages)
        assert response.tool_prediction
        assert response.input_tokens > 0
        assert response.output_tokens > 0
        assert response.cache_write_tokens > 0
        previous_cache_write_tokens = response.cache_write_tokens
        assert response.cache_read_tokens == 0
        expected_total_tokens = (
            response.input_tokens
            + response.output_tokens
            + response.cache_write_tokens
        )
        assert response.total_tokens == expected_total_tokens
        assert response.input_cost > 0
        assert response.output_cost > 0
        assert response.cache_write_cost > 0
        assert response.cache_read_cost == 0
        expected_total_cost = (
            response.input_cost
            + response.output_cost
            + response.cache_write_cost
        )
        assert response.total_cost == expected_total_cost

        # second run should result in a cache-hit
        messages=[
            user_message("Search for expensive italian restaurants in New York?"),
        ]
        response = client(messages=messages)
        assert response.tool_prediction
        assert response.input_tokens > 0
        assert response.output_tokens > 0
        assert response.cache_write_tokens == 0
        assert response.cache_read_tokens == previous_cache_write_tokens
        expected_total_tokens = (
            response.input_tokens
            + response.output_tokens
            + response.cache_read_tokens
        )
        assert response.total_tokens == expected_total_tokens
        assert response.input_cost > 0
        assert response.output_cost > 0
        assert response.cache_write_cost == 0
        assert response.cache_read_cost > 0
        expected_total_cost = (
            response.input_cost
            + response.output_cost
            + response.cache_read_cost
        )
        assert response.total_cost == expected_total_cost


@pytest.mark.integration  # these tests make API calls
@pytest.mark.skipif(os.getenv('ANTHROPIC_API_KEY') is None, reason="ANTHROPIC_API_KEY is not set")
class TestAnthropicReasoning:
    """Test the Anthropic Wrapper with Reasoning."""

    @pytest.mark.parametrize(
        "reasoning_effort",
        [ReasoningEffort.LOW, ReasoningEffort.MEDIUM, ReasoningEffort.HIGH],
    )
    def test__Anthropic_instantiate__with_reasoning_effort(self, reasoning_effort: ReasoningEffort):  # noqa: E501
        model = Client.instantiate(
            client_type=RegisteredClients.ANTHROPIC,
            model_name=ANTRHOPIC_TEST_THINKING_MODEL,
            reasoning_effort=reasoning_effort,
            max_tokens=4000,
        )
        assert isinstance(model, Anthropic)
        assert model.model_parameters.get('thinking') is not None
        assert model.model_parameters['thinking']['type'] == 'enabled'
        assert model.model_parameters['thinking']['budget_tokens']
        assert model.model_parameters['max_tokens'] > model.model_parameters['thinking']['budget_tokens']  # noqa: E501

    def test__Anthropic_instantiate__with_thinking_budget(self):
        model = Client.instantiate(
            client_type=RegisteredClients.ANTHROPIC,
            model_name=ANTRHOPIC_TEST_THINKING_MODEL,
            thinking_budget_tokens=12000,
            max_tokens=4000,
        )
        assert isinstance(model, Anthropic)
        assert model.model_parameters.get('thinking') is not None
        assert model.model_parameters['thinking']['type'] == 'enabled'
        assert model.model_parameters['thinking']['budget_tokens'] == 12000
        assert model.model_parameters['max_tokens'] > model.model_parameters['thinking']['budget_tokens']  # noqa: E501

    @pytest.mark.asyncio
    async def test__Anthropic__with_thinking__reasoning_effort(self):
        """Test that the extended thinking chunks have the correct content types."""
        model = Client.instantiate(
            client_type=RegisteredClients.ANTHROPIC,
            model_name=ANTRHOPIC_TEST_THINKING_MODEL,
            reasoning_effort=ReasoningEffort.LOW,
        )
        # Use a prompt that should trigger some thinking content
        has_thinking_content = False
        has_text_content = False

        messages = [user_message("What is 1 + 2 + (3 * 4) + (5 * 6)?")]
        async for response in model.stream(messages=messages):
            if isinstance(response, TextChunkEvent):
                has_text_content = True
            elif isinstance(response, ThinkingChunkEvent):
                has_thinking_content = True
            elif isinstance(response, TextResponse):
                # Check that summary contains both thinking and answer
                assert "45" in response.response

        assert has_thinking_content, "No thinking content was generated"
        assert has_text_content, "No text content was generated"

    @pytest.mark.asyncio
    async def test__Anthropic__with_thinking__thinking_budget_tokens(self):
        """Test that the extended thinking chunks have the correct content types."""
        model = Client.instantiate(
            client_type=RegisteredClients.ANTHROPIC,
            model_name=ANTRHOPIC_TEST_THINKING_MODEL,
            thinking_budget_tokens=2000,
        )
        # Use a prompt that should trigger some thinking content
        has_thinking_content = False
        has_text_content = False
        has_summary = False

        messages = [user_message("What is 1 + 2 + (3 * 4) + (5 * 6)?")]
        async for response in model.stream(messages=messages):
            if isinstance(response, TextChunkEvent):
                has_text_content = True
            elif isinstance(response, ThinkingChunkEvent):
                has_thinking_content = True
            elif isinstance(response, TextResponse):
                # Check that summary contains both thinking and answer
                has_summary = True
                assert "45" in response.response

        assert has_thinking_content, "No thinking content was generated"
        assert has_text_content, "No text content was generated"
        assert has_summary, "No summary was generated"

    @pytest.mark.asyncio
    async def test__Anthropic__with_thinking__temperature(self):
        """
        Test that the extended thinking works when temperature is set.

        From docs: "Thinking isn't compatible with temperature, top_p, or top_k modifications as
        well as forced tool use."

        Make sure it doesn't crash.
        """
        model = Client.instantiate(
            client_type=RegisteredClients.ANTHROPIC,
            model_name=ANTRHOPIC_TEST_THINKING_MODEL,
            reasoning_effort=ReasoningEffort.LOW,
            temperature=0.5,
        )
        # Use a prompt that should trigger some thinking content
        has_thinking_content = False
        has_text_content = False

        messages = [user_message("What is 1 + 2 + (3 * 4) + (5 * 6)?")]
        async for response in model.stream(messages=messages):
            if isinstance(response, TextChunkEvent):
                has_text_content = True
            elif isinstance(response, ThinkingChunkEvent):
                has_thinking_content = True
            elif isinstance(response, TextResponse):
                # Check that summary contains both thinking and answer
                assert "45" in response.response

        assert has_thinking_content, "No thinking content was generated"
        assert has_text_content, "No text content was generated"

    @pytest.mark.asyncio
    async def test__Anthropic__with_thinking__test_redacted_thinking(self):
        """Test that the extended thinking chunks have the correct content types."""
        model = Client.instantiate(
            client_type=RegisteredClients.ANTHROPIC,
            model_name=ANTRHOPIC_TEST_THINKING_MODEL,
            thinking_budget_tokens=2000,
        )
        # Use a prompt that should trigger some thinking content
        has_redacted_thinking_content = False
        has_thinking_content = False
        has_text_content = False
        has_summary = False

        # test redacted thinking:
        # https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#example-working-with-redacted-thinking-blocks
        messages = [user_message("ANTHROPIC_MAGIC_STRING_TRIGGER_REDACTED_THINKING_46C9A13E193C177646C7398A98432ECCCE4C1253D5E2D82641AC0E52CC2876CB")]  # noqa: E501
        async for response in model.stream(messages=messages):
            if isinstance(response, TextChunkEvent):
                has_text_content = True
            elif isinstance(response, ThinkingChunkEvent):
                if response.is_redacted:
                    has_redacted_thinking_content = True
                    assert response.content
                else:
                    has_thinking_content = True
            elif isinstance(response, TextResponse):
                has_summary = True
                assert response.response

        assert has_redacted_thinking_content, "No redacted thinking content was generated"
        assert not has_thinking_content, "Thinking content was generated"
        assert has_text_content, "No text content was generated"
        assert has_summary, "No summary was generated"

    @pytest.mark.asyncio
    async def test__Anthropic__without_thinking__verify_no_thinking_events(self):
        """Test that the extended thinking chunks have the correct content types."""
        model = Client.instantiate(
            client_type=RegisteredClients.ANTHROPIC,
            model_name=ANTRHOPIC_TEST_THINKING_MODEL,
        )
        # Use a prompt that should trigger some thinking content
        has_thinking_content = False
        has_text_content = False
        has_summary = False

        messages = [user_message("What is 1 + 2 + (3 * 4) + (5 * 6)?")]
        async for response in model.stream(messages=messages):
            if isinstance(response, TextChunkEvent):
                has_text_content = True
            elif isinstance(response, ThinkingChunkEvent):
                has_thinking_content = True
            elif isinstance(response, TextResponse):
                # Check that summary contains both thinking and answer
                has_summary = True

        assert not has_thinking_content, "Thinking content was generated"
        assert has_text_content, "No text content was generated"
        assert has_summary, "No summary was generated"
