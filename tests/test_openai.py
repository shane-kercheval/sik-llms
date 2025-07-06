"""Test the OpenAI Wrapper."""
import asyncio
import random
from time import perf_counter, sleep
from faker import Faker
import pytest
from dotenv import load_dotenv
from sik_llms import (
    Client,
    ReasoningEffort,
    create_client,
    system_message,
    user_message,
    Tool,
    Parameter,
    RegisteredClients,
    OpenAI,
    TextChunkEvent,
    TextResponse,
    OpenAITools,
)
from sik_llms.models_base import ImageContent, ImageSourceType, assistant_message
from sik_llms.openai import SUPPORTED_OPENAI_MODELS, _convert_messages
from tests.conftest import OPENAI_TEST_MODEL, OPENAI_TEST_REASONING_MODEL

load_dotenv()


def test__registration__openai():
    assert Client.is_registered(RegisteredClients.OPENAI)


def test__registration__openai_tools():
    assert Client.is_registered(RegisteredClients.OPENAI_TOOLS)


@pytest.mark.asyncio
class TestOpenAIRegistration:
    """Test the OpenAI Completion Wrapper registration."""

    @pytest.mark.stochastic(samples=10, threshold=0.8)
    async def test__openai__instantiate(self):
        client = Client.instantiate(
            client_type=RegisteredClients.OPENAI,
            model_name=OPENAI_TEST_MODEL,
        )
        assert isinstance(client, OpenAI)
        assert client.model == OPENAI_TEST_MODEL
        assert client.client is not None
        assert client.client.api_key
        assert client.client.api_key != 'None'
        assert client.client.base_url

        # call async/stream method
        responses = []
        async for response in client.stream(messages=[user_message("What is the capital of France?")]):  # noqa: E501
            if isinstance(response, TextChunkEvent):
                responses.append(response)

        assert len(responses) > 0
        assert 'Paris' in ''.join([response.content for response in responses])

        # call non-async method
        response = client(messages=[user_message("What is the capital of France?")])
        assert isinstance(response, TextResponse)
        assert 'Paris' in response.response

    async def test__openai__compatible_server_instantiate___parameters_max_completion_tokens(self):
        model_config = {
            'server_url': 'http://localhost:8000',
            'max_completion_tokens': 10,
        }
        client = Client.instantiate(
            client_type=RegisteredClients.OPENAI,
            model_name='openai-compatible-server',
            **model_config,
        )
        assert isinstance(client, OpenAI)
        assert client.model == 'openai-compatible-server'
        assert client.model_parameters['max_completion_tokens'] == 10

    async def test__openai__tools_instantiate(self):
        client = Client.instantiate(
            client_type=RegisteredClients.OPENAI_TOOLS,
            model_name=OPENAI_TEST_MODEL,
            tools=[
                Tool(
                    name='get_weather',
                    description='Get the weather for a location.',
                    parameters=[
                        Parameter(
                            name='location',
                            param_type=str,
                            required=True,
                            description='The city and country for weather info.',
                        ),
                    ],
                ),
            ],
        )
        assert isinstance(client, OpenAITools)
        assert client.model == OPENAI_TEST_MODEL
        assert client.client is not None
        assert client.client.api_key
        assert client.client.api_key != 'None'
        assert client.client.base_url
        assert client


class TestConvertMessages:
    """Test the _convert_messages function."""

    def test__convert_messages(self):
        converted = _convert_messages(messages=[])
        assert converted == []

        converted = _convert_messages(messages=[
            system_message("You are a helpful assistant."),
        ])
        assert converted == [
            {"role": "system", "content": "You are a helpful assistant."},
        ]

        converted = _convert_messages(messages=[
            system_message("You are a helpful assistant."),
            user_message("What is the capital of France?"),
        ])
        assert converted == [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ]

    def test__converted_messages__cache_content__no_system_message(self):
        messages = [
            user_message("What is the capital of France?"),
        ]
        expected = [
            {"role": "user", "content": "<cache_content>"},
            {"role": "user", "content": "What is the capital of France?"},
        ]
        # list
        converted = _convert_messages(
            messages=messages,
            cache_content=["<cache_content>"],
        )
        assert converted == expected
        # str
        converted = _convert_messages(
            messages=messages,
            cache_content="<cache_content>",
        )
        assert converted == expected

        converted = _convert_messages(
            messages=[
                user_message("What is the capital of France?"),
            ],
            cache_content=["<cache_content 1>", "<cache_content 2>"],
        )
        assert converted == [
            {"role": "user", "content": "<cache_content 1>"},
            {"role": "user", "content": "<cache_content 2>"},
            {"role": "user", "content": "What is the capital of France?"},
        ]

    def test__converted_messages__cache_content__no_user_message(self):
        expected = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "<cache_content>"},
        ]
        messages = [
            system_message("You are a helpful assistant."),
        ]
        # list
        converted = _convert_messages(
            messages=messages,
            cache_content=["<cache_content>"],
        )
        assert converted == expected
        # str
        converted = _convert_messages(
            messages=messages,
            cache_content="<cache_content>",
        )
        assert converted == expected

        converted = _convert_messages(
            messages=[
                system_message("You are a helpful assistant."),
            ],
            cache_content=["<cache_content 1>", "<cache_content 2>"],
        )
        assert converted == [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "<cache_content 1>"},
            {"role": "user", "content": "<cache_content 2>"},
        ]

    def test__converted_messages__cache_content__user_and_system_message(self):
        messages = [
            system_message("You are a helpful assistant."),
            user_message("What is the capital of France?"),
        ]
        expected = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "<cache_content>"},
            {"role": "user", "content": "What is the capital of France?"},
        ]
        converted = _convert_messages(
            messages=messages,
            cache_content=["<cache_content>"],
        )
        assert converted == expected
        converted = _convert_messages(
            messages=messages,
            cache_content="<cache_content>",
        )
        assert converted == expected

        converted = _convert_messages(
            messages=[
                system_message("You are a helpful assistant."),
                system_message("system message 2."),
                user_message("What is the capital of France?"),
                assistant_message("assistant message 1."),
                user_message("assistant message 2."),
            ],
            cache_content=["<cache_content 1>", "<cache_content 2>"],
        )
        assert converted == [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "system", "content": "system message 2."},
            {"role": "user", "content": "<cache_content 1>"},
            {"role": "user", "content": "<cache_content 2>"},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "assistant message 1."},
            {"role": "user", "content": "assistant message 2."},
        ]

    def test_basic_messages(self):
        """Test basic message conversion without images or caching."""
        converted = _convert_messages(messages=[])
        assert converted == []

        converted = _convert_messages(messages=[
            system_message("You are a helpful assistant."),
        ])
        assert converted == [
            {"role": "system", "content": "You are a helpful assistant."},
        ]

        converted = _convert_messages(messages=[
            system_message("You are a helpful assistant."),
            user_message("What is the capital of France?"),
        ])
        assert converted == [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ]

    def test_image_conversion_url(self):
        """Test conversion of messages containing URL images."""
        image = ImageContent.from_url("https://example.com/image.jpg")
        messages = [
            user_message([
                "What's in this image?",
                image,
            ]),
        ]

        converted = _convert_messages(messages=messages)
        assert converted == [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What's in this image?",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image.jpg"},
                },
            ],
        }]

    def test_image_conversion_base64(self):
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

        converted = _convert_messages(messages=messages)
        assert converted == [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What's in this image?",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,base64data"},
                },
            ],
        }]

    def test_multiple_images_and_text(self):
        """Test conversion of messages containing multiple images and text segments."""
        image1 = ImageContent.from_url("https://example.com/image1.jpg")
        image2 = ImageContent.from_url("https://example.com/image2.jpg")
        messages = [
            user_message([
                "Here are two images.",
                image1,
                "This is the first image above and second image below.",
                image2,
                "What are the differences between them?",
            ]),
        ]

        converted = _convert_messages(messages=messages)
        expected_content = [
            {"type": "text", "text": "Here are two images."},
            {"type": "image_url", "image_url": {"url": "https://example.com/image1.jpg"}},
            {"type": "text", "text": "This is the first image above and second image below."},
            {"type": "image_url", "image_url": {"url": "https://example.com/image2.jpg"}},
            {"type": "text", "text": "What are the differences between them?"},
        ]
        assert converted == [{"role": "user", "content": expected_content}]

    def test_cache_content_basic(self):
        """Test cache content insertion with basic messages."""
        messages = [
            user_message("What is the capital of France?"),
        ]
        cache_content = "Cached information about France."

        converted = _convert_messages(messages=messages, cache_content=cache_content)
        assert converted == [
            {"role": "user", "content": "Cached information about France."},
            {"role": "user", "content": "What is the capital of France?"},
        ]

    def test_cache_content_with_system_message(self):
        """Test cache content insertion with system messages."""
        messages = [
            system_message("You are a helpful assistant."),
            user_message("What is the capital of France?"),
        ]
        cache_content = ["Cached info 1", "Cached info 2"]

        converted = _convert_messages(messages=messages, cache_content=cache_content)
        assert converted == [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Cached info 1"},
            {"role": "user", "content": "Cached info 2"},
            {"role": "user", "content": "What is the capital of France?"},
        ]

    def test_cache_and_images_combined(self):
        """Test combining both cache content and images in messages."""
        image = ImageContent.from_url("https://example.com/image.jpg")
        messages = [
            system_message("You are a helpful assistant."),
            user_message([
                "What's in this image?",
                image,
            ]),
        ]
        cache_content = "Background information about images."

        converted = _convert_messages(messages=messages, cache_content=cache_content)
        assert converted == [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Background information about images."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
                ],
            },
        ]

    def test_complex_conversation_with_images_and_cache(self):
        """Test a complex conversation with multiple messages, images, and cache content."""
        image1 = ImageContent.from_url("https://example.com/image1.jpg")
        image2 = ImageContent.from_url("https://example.com/image2.jpg")
        messages = [
            system_message("You are an image analysis assistant."),
            user_message(["Here's the first image:", image1]),
            assistant_message("I see a landscape photo."),
            user_message(["And here's another:", image2, "Compare them."]),
        ]
        cache_content = ["Image analysis guidelines.", "Additional context."]

        converted = _convert_messages(messages=messages, cache_content=cache_content)
        assert len(converted) == 6  # system + 2 cache + user + assistant + user
        assert converted[0]["role"] == "system"
        assert converted[1]["role"] == "user"  # First cache message
        assert converted[2]["role"] == "user"  # Second cache message
        assert isinstance(converted[3]["content"], list)  # First image message
        assert converted[4]["role"] == "assistant"
        assert isinstance(converted[5]["content"], list)  # Second image message

    def test_message_content_stripping(self):
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

        converted = _convert_messages(messages=messages, cache_content=cache_content)
        assert converted[0]["content"] == "System message with spaces"
        assert converted[1]["content"] == "Cached content with spaces"
        assert converted[2]["content"][0]["text"] == "Text with spaces"
        assert converted[2]["content"][2]["text"] == "Text with newlines"


class TestReasoningModelsDoNotSetCertainParameters:
    """When using reasoning models, certain parameters should not be set."""

    def test__check_parameters_for_none_reasoning(self):
        client = OpenAI(
            model_name=OPENAI_TEST_MODEL,
            reasoning_effort=None,
            temperature=0.5,
            top_p=0.9,
        )
        assert client.reasoning_effort is None
        assert 'temperature' in client.model_parameters
        assert 'top_p' in client.model_parameters

    def test__does_not_set_params__reasoning_model(self):
        client = OpenAI(
            model_name=OPENAI_TEST_REASONING_MODEL,
            # reasoning model but no reasoning effort set
            reasoning_effort=None,
            temperature=0.5,
            top_p=0.9,
            logprobs=True,
        )
        assert client.reasoning_effort is None
        assert 'temperature' not in client.model_parameters
        assert 'top_p' not in client.model_parameters
        assert 'logprobs' not in client.model_parameters

    @pytest.mark.parametrize('reasoning_effort', [None, ReasoningEffort.LOW])
    def test__does_not_set_params__reasoning_effort(self, reasoning_effort: ReasoningEffort | None):  # noqa: E501
        client = OpenAI(
            model_name=OPENAI_TEST_REASONING_MODEL if reasoning_effort else OPENAI_TEST_MODEL,
            reasoning_effort=reasoning_effort,
            temperature=0.5,
            top_p=0.9,
            logprobs=True,
        )
        assert client.reasoning_effort == reasoning_effort
        assert 'temperature' not in client.model_parameters if reasoning_effort else 'temperature' in client.model_parameters  # noqa: E501
        assert 'top_p' not in client.model_parameters if reasoning_effort else 'top_p' in client.model_parameters  # noqa: E501
        assert 'logprobs' not in client.model_parameters if reasoning_effort else 'logprobs' in client.model_parameters  # noqa: E501


@pytest.mark.asyncio
class TestOpenAI:
    """Test the OpenAI Completion Wrapper."""

    async def test__all_models(self):
        """Test all models concurrently for better performance."""
        async def test_single_model(model_name: str) -> TextResponse:
            """Test a single model and return results."""
            client = OpenAI(model_name=model_name)
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
        model_names = list(SUPPORTED_OPENAI_MODELS.keys())
        results = await asyncio.gather(
            *(test_single_model(model) for model in model_names),
            return_exceptions=True,
        )
        assert len(results) == len(model_names)
        assert all(isinstance(result, TextResponse) for result in results)



    @pytest.mark.stochastic(samples=10, threshold=0.8)
    async def test__async_openai(self):
        # Create an instance of the wrapper
        client = OpenAI(model_name=OPENAI_TEST_MODEL)
        messages = [
            system_message("You are a helpful assistant."),
            user_message("What is the capital of France?"),
        ]
        chunks = []
        summary = None
        async for c in client.stream(messages=messages):
            if isinstance(c, TextChunkEvent):
                chunks.append(c)
            elif isinstance(c, TextResponse):
                summary = c
        response = ''.join([chunk.content for chunk in chunks])
        assert 'Paris' in response
        assert isinstance(summary, TextResponse)
        assert summary.input_tokens > 0
        assert summary.output_tokens > 0
        assert summary.total_tokens > 0
        assert summary.input_cost > 0
        assert summary.output_cost > 0
        assert summary.total_cost > 0
        assert summary.duration_seconds > 0

    async def test__openai_logprobs(self):
        # Create an instance of the wrapper
        client = OpenAI(
            model_name=OPENAI_TEST_MODEL,
            logprobs=True,
        )
        messages = [
            system_message("You are a helpful assistant."),
            user_message("What is the capital of France?"),
        ]
        chunks = []
        async for response in client.stream(messages=messages):
            if isinstance(response, TextChunkEvent):
                chunks.append(response)
        assert all(chunk.logprob is not None for chunk in chunks)

    async def test_concurrent_performance(self):
        # This test will use the actual API, so consider using it sparingly
        client = OpenAI(model_name=OPENAI_TEST_MODEL)
        num_requests = 20
        messages = [{"role": "user", "content": "Say hello in one word."}]

        async def execute_request(i: int):  # noqa: ANN202
            start = perf_counter()
            result = None
            async for chunk in client.stream(messages=messages):
                if hasattr(chunk, 'content') and isinstance(chunk.content, str):
                    result = chunk
            end = perf_counter()
            return {"index": i, "time": end - start, "result": result}

        # Execute all requests asynchronously/concurrently
        start_time = perf_counter()
        tasks = [execute_request(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)
        actual_time = perf_counter() - start_time

        times = [r["time"] for r in results]
        sequential_time = sum(times)

        assert len(results) == num_requests
        # The actual time should be significantly less than sequential execution
        assert actual_time < (sequential_time / 2)


@pytest.mark.asyncio
@pytest.mark.integration  # these tests make API calls
class TestOpenAIReasoning:
    """Test the OpenAI Reasoning / ReasoningEffort."""

    async def test__openai__reasoning(self):
        client = create_client(
            model_name=OPENAI_TEST_REASONING_MODEL,
            reasoning_effort=ReasoningEffort.LOW,
        )
        response = client(messages=[user_message("What is 1 + 2 + (3 * 4) + (5 * 6)?")])
        assert isinstance(response, TextResponse)
        assert '45' in response.response
        assert response.input_tokens > 0
        assert response.output_tokens > 0
        assert response.total_tokens > 0
        assert response.input_cost > 0
        assert response.output_cost > 0
        assert response.total_cost > 0
        assert response.duration_seconds > 0
        response = client(messages=[user_message("What is the capital of France?")])
        assert 'Paris' in response.response

    async def test__openai__test_reasoning_model__without_reasonsing_effort_set(self):
        """
        Logprobs are not supported for reasoning models so let's test that we don't set it. In
        general, let's test that the reasoning model works without setting reasonsing.
        """
        client = create_client(
            model_name=OPENAI_TEST_REASONING_MODEL,
            reasoning_effort=None,
        )
        response = client(messages=[user_message("What is the capital of France?")])
        assert isinstance(response, TextResponse)
        assert 'Paris' in response.response
        assert response.input_tokens > 0
        assert response.output_tokens > 0
        assert response.total_tokens > 0
        assert response.input_cost > 0
        assert response.output_cost > 0
        assert response.total_cost > 0
        assert response.duration_seconds > 0


@pytest.mark.skip_ci  # caching sporadically fails; let's skip this in CI
@pytest.mark.integration  # these tests make API calls
class TestOpenAICaching:
    """
    Test the caching functionality of the OpenAI Wrapper.

    Not tested with Anthropic because each provider has different caching behavior. Anthropic has
    mechanisms to specify caching behavior in the request, but OpenAI caches automatically.
    Additionally, Anthropic differentiates between cache hits and misses, while OpenAI does not.

    https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#why-am-i-seeing-the-error-attributeerror-beta-object-has-no-attribute-prompt-caching-in-python
    https://github.com/anthropics/anthropic-cookbook/blob/main/misc/prompt_caching.ipynb
    """

    @pytest.mark.parametrize('use_init', [True, False])
    def test__OpenAI__caching__expect_cache_miss_then_hit(self, use_init: bool):
        """
        Tests that the first call results in a cache-miss and the second call results in a
        cache-hit. The tokens written to the cache should be read on the second call.
        """
        length = random.randint(15_000, 20_000)
        first_word = Faker().words(nb=1)[0]
        cache_content = f"{first_word} is the first word. {Faker().text(max_nb_chars=length)}"

        client = create_client(
            model_name=OPENAI_TEST_MODEL,
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
                user_message(cache_content),
                user_message("What is the first word of this text? Only output the first word."),
            ]

        # first run should result in a cache-miss & write
        response = client(messages=messages)
        assert response.response
        assert response.input_tokens > 0
        assert response.output_tokens > 0
        # first run should result in a cache-miss
        assert response.cache_read_tokens == 0
        expected_total_tokens = (
            response.input_tokens
            + response.output_tokens
        )
        assert response.total_tokens == expected_total_tokens
        assert response.input_cost > 0
        assert response.output_cost > 0
        # first run should result in a cache-miss
        assert response.cache_read_cost == 0
        expected_total_cost = (
            response.input_cost
            + response.output_cost
        )
        assert response.total_cost == expected_total_cost

        sleep(0.5)  # sporadicly failing; i'm guessing it needs time to write to cache
        # second run should result in a cache-hit
        response = client(messages=messages)
        assert response.response
        assert response.input_tokens > 0
        assert response.output_tokens > 0
        # now there should be a cache hit
        assert response.cache_read_tokens > 0
        expected_total_tokens = (
            response.input_tokens
            + response.output_tokens
            + response.cache_read_tokens
        )
        assert response.total_tokens == expected_total_tokens
        assert response.input_cost > 0
        assert response.output_cost > 0
        # now there should be a cache hit
        assert response.cache_read_cost > 0
        expected_total_cost = (
            response.input_cost
            + response.output_cost
            + response.cache_read_cost
        )
        assert response.total_cost == expected_total_cost

    def test__OpenAI__caching__with_tools(self, complex_weather_tool: Tool, restaurant_tool: Tool):
        # we need to make the text long enough to trigger the cache
        text = "[THE FOLLOWING IS TEST DATA FOR TESTING PURPOSES ONLY; **INGNORE THE FOLLOWING TEXT**]"  # noqa: E501
        text += Faker().text(max_nb_chars=15_000)
        complex_weather_tool.parameters[-1].description = "\n\n" + text

        client = OpenAITools(
            model_name=OPENAI_TEST_MODEL,
            tools=[complex_weather_tool, restaurant_tool],
        )
        messages=[
            user_message("What's the weather like in Tokyo in Celsius with forecast?"),
        ]
        response = client(messages=messages)
        assert response.tool_prediction
        assert response.input_tokens > 0
        assert response.output_tokens > 0
        # first run should result in a cache-miss
        assert response.cache_read_tokens == 0
        expected_total_tokens = (
            response.input_tokens
            + response.output_tokens
        )
        assert response.total_tokens == expected_total_tokens
        assert response.input_cost > 0
        assert response.output_cost > 0
        # first run should result in a cache-miss
        assert response.cache_read_cost == 0
        expected_total_cost = (
            response.input_cost
            + response.output_cost
        )
        assert response.total_cost == expected_total_cost

        sleep(0.5)  # sporadicly failing; i'm guessing it needs time to write to cache
        # second run should result in a cache-hit
        messages=[
            user_message("Search for expensive italian restaurants in New York?"),
        ]
        response = client(messages=messages)
        assert response.tool_prediction
        assert response.input_tokens > 0
        assert response.output_tokens > 0
        # now there should be a cache hit
        assert response.cache_read_tokens > 0
        expected_total_tokens = (
            response.input_tokens
            + response.output_tokens
            + response.cache_read_tokens
        )
        assert response.total_tokens == expected_total_tokens
        assert response.input_cost > 0
        assert response.output_cost > 0
        # now there should be a cache hit
        assert response.cache_read_cost > 0
        expected_total_cost = (
            response.input_cost
            + response.output_cost
            + response.cache_read_cost
        )
        assert response.total_cost == expected_total_cost
