"""Tests for the models_base module."""
import pytest
from pathlib import Path
import base64
from unittest.mock import mock_open, patch
from sik_llms import (
    Client,
    RegisteredClients,
    system_message,
    assistant_message,
    user_message,
    ImageContent,
    ImageSourceType,
)
from sik_llms.models_base import Parameter


class TestMesssages:
    """Tests for the message creation functions."""

    def test_system_message(self):
        message = system_message('test')
        assert message == {'role': 'system', 'content': 'test'}

    def test_system_message_with_kwargs(self):
        message = system_message('test', foo='bar')
        assert message == {'role': 'system', 'content': 'test', 'foo': 'bar'}

    def test_user_message(self):
        message = user_message('test')
        assert message == {'role': 'user', 'content': 'test'}

    def test_assistant_message(self):
        message = assistant_message('test')
        assert message == {'role': 'assistant', 'content': 'test'}

    def test_simple_text_message(self):
        """Test creating a message with just text."""
        text = "Hello, how are you?"
        message = user_message(text)

        assert message == {
            'role': 'user',
            'content': 'Hello, how are you?',
        }

    def test_text_message_with_whitespace(self):
        """Test that whitespace is stripped from text messages."""
        text = "  Hello, how are you?  \n"
        message = user_message(text)

        assert message == {
            'role': 'user',
            'content': 'Hello, how are you?',
        }

    def test_empty_text_message(self):
        """Test handling empty text message."""
        text = ""
        message = user_message(text)

        assert message == {
            'role': 'user',
            'content': '',
        }

    def test_message_with_single_image(self):
        """Test creating a message with one image."""
        image = ImageContent.from_url("https://example.com/image.jpg")
        message = user_message([image])

        assert message == {
            'role': 'user',
            'content': [image],
        }

    def test_message_with_text_and_image(self):
        """Test creating a message with text and image."""
        image = ImageContent.from_url("https://example.com/image.jpg")
        content = ["What's in this image?", image]
        message = user_message(content)

        assert message == {
            'role': 'user',
            'content': content,
        }

    def test_message_with_multiple_images(self):
        """Test creating a message with multiple images."""
        image1 = ImageContent.from_url("https://example.com/image1.jpg")
        image2 = ImageContent.from_url("https://example.com/image2.jpg")
        text = "Compare these images"
        content = [text, image1, image2]
        message = user_message(content)

        assert message == {
            'role': 'user',
            'content': content,
        }

    def test_message_with_complex_content(self):
        """Test creating a message with text before, between, and after images."""
        image1 = ImageContent.from_url("https://example.com/image1.jpg")
        image2 = ImageContent.from_url("https://example.com/image2.jpg")
        content = [
            "Here are two images.",
            image1,
            "This is the first image above and second image below.",
            image2,
            "What are the differences between them?",
        ]
        message = user_message(content)

        assert message == {
            'role': 'user',
            'content': content,
        }

    def test_message_with_whitespace_in_list(self):
        """Test that whitespace is stripped from text items in content list."""
        image = ImageContent.from_url("https://example.com/image.jpg")
        content = ["  First text  ", image, "\nSecond text\n"]
        message = user_message(content)

        assert message == {
            'role': 'user',
            'content': ["First text", image, "Second text"],
        }

    def test_invalid_content_type(self):
        """Test handling invalid content types."""
        with pytest.raises(TypeError):
            user_message(123)

    def test_invalid_list_content(self):
        """Test handling invalid items in content list."""
        with pytest.raises(TypeError):
            user_message([1, 2, 3])

    def test_nested_lists(self):
        """Test that nested lists are not allowed."""
        with pytest.raises(ValueError):  # noqa: PT011
            user_message([["nested"], "content"])

    def test_none_content(self):
        """Test handling None content."""
        with pytest.raises(TypeError):
            user_message(None)


class MockClient(Client):  # noqa: D101
    def __init__(self, model_name: str, **kwargs):  # noqa: ANN003
        self.model_name = model_name
        self.kwargs = kwargs

    async def stream(self, messages):  # noqa: ANN001
        # Simple implementation for testing
        return messages


@Client.register('MockClient')
class RegisteredMockClient(MockClient):  # noqa: D101
    pass


class TestClientRegistration:
    """Tests for the Client registration system."""

    def test_client_registration(self):
        # Test registration with string
        assert Client.is_registered('MockClient')
        # Test registration with enum
        assert Client.is_registered(RegisteredClients.OPENAI)
        # Test unregistered client
        assert not Client.is_registered('UnknownClient')

    def test_client_instantiation(self):
        # Test instantiation with string registration
        client = Client.instantiate(
            'MockClient',
            model_name='test-model',
            temperature=0.7,
            max_tokens=1000,
        )
        assert isinstance(client, RegisteredMockClient)
        assert client.model_name == 'test-model'
        assert client.kwargs['temperature'] == 0.7
        assert client.kwargs['max_tokens'] == 1000

        # Test instantiation with enum registration
        client = Client.instantiate(
            RegisteredClients.OPENAI,
            model_name='gpt-4o',
            max_tokens=100,
        )
        assert client.model == 'gpt-4o'
        # openai deprecated max_tokens in favor of max_completion_tokens
        assert client.model_parameters['max_completion_tokens'] == 100

    def test_model_kwargs_not_modified(self):
        original_kwargs = {'temperature': 0.7, 'max_tokens': 100}
        kwargs_copy = original_kwargs.copy()

        client = Client.instantiate(
            'MockClient',
            model_name='test-model',
            **original_kwargs,
        )

        # Verify original kwargs weren't modified
        assert original_kwargs == kwargs_copy

        # Verify client has correct kwargs
        assert client.kwargs['temperature'] == 0.7
        assert client.kwargs['max_tokens'] == 100
        assert client.model_name == 'test-model'

        # Modify the original kwargs and verify client kwargs are unchanged
        original_kwargs['temperature'] = 0.5
        original_kwargs['max_tokens'] = 200
        assert client.kwargs['temperature'] == 0.7
        assert client.kwargs['max_tokens'] == 100


class TestImageContent:
    """Tests for the ImageContent class."""

    def test_from_url(self):
        """Test creating ImageContent from URL."""
        url = "https://example.com/image.jpg"
        image = ImageContent.from_url(url)
        assert image.source_type == ImageSourceType.URL
        assert image.data == url
        assert image.media_type is None

    @pytest.mark.parametrize(('file_extension', 'expected_media_type'), [
        ("jpg", "image/jpeg"),
        ("jpeg", "image/jpeg"),
        ("png", "image/png"),
        ("gif", "image/gif"),
    ])
    def test_from_path(self, file_extension: str, expected_media_type: str):
        """Test creating ImageContent from file path with different extensions."""
        test_data = b"fake image data"
        expected_base64 = base64.b64encode(test_data).decode("utf-8")
        mock_path = Path(f"test_image.{file_extension}")
        with patch("builtins.open", mock_open(read_data=test_data)):
            image = ImageContent.from_path(mock_path)
        assert image.source_type == ImageSourceType.BASE64
        assert image.data == expected_base64
        assert image.media_type == expected_media_type

    def test_from_path_string(self):
        """Test creating ImageContent from string path."""
        test_data = b"fake image data"
        expected_base64 = base64.b64encode(test_data).decode("utf-8")
        with patch("builtins.open", mock_open(read_data=test_data)):
            image = ImageContent.from_path("test_image.jpg")
        assert image.source_type == ImageSourceType.BASE64
        assert image.data == expected_base64
        assert image.media_type == "image/jpeg"

    def test_from_bytes(self):
        """Test creating ImageContent from bytes."""
        test_data = b"fake image data"
        media_type = "image/jpeg"
        expected_base64 = base64.b64encode(test_data).decode("utf-8")
        image = ImageContent.from_bytes(test_data, media_type)
        assert image.source_type == ImageSourceType.BASE64
        assert image.data == expected_base64
        assert image.media_type == media_type

    def test_from_path_file_not_found(self):
        """Test error handling when file not found."""
        with pytest.raises(FileNotFoundError):
            ImageContent.from_path("nonexistent_image.jpg")

    def test_model_validation(self):
        """Test Pydantic model validation."""
        # Valid data
        data = {
            "source_type": ImageSourceType.URL,
            "data": "https://example.com/image.jpg",
            "media_type": "image/jpeg",
        }
        image = ImageContent(**data)
        assert image.source_type == ImageSourceType.URL
        assert image.data == "https://example.com/image.jpg"
        assert image.media_type == "image/jpeg"
        # Invalid source_type
        with pytest.raises(ValueError):  # noqa: PT011
            ImageContent(
                source_type="invalid",
                data="https://example.com/image.jpg",
            )

    @pytest.mark.parametrize(('test_input', 'expected'), [
        (b"test data", "dGVzdCBkYXRh"),  # Known base64 encoding
        (b"", ""),  # Empty bytes
        (b"123", "MTIz"),  # Numbers
        (b"!@#$%", "IUAjJCU="),  # Special characters
    ])
    def test_base64_encoding(self, test_input: bytes, expected: str):
        """Test base64 encoding with various inputs."""
        image = ImageContent.from_bytes(test_input, "image/jpeg")
        assert image.data == expected

    def test_large_file_handling(self):
        """Test handling of large file (simulate with mock)."""
        large_data = b"x" * 1024 * 1024  # 1MB of data
        expected_base64 = base64.b64encode(large_data).decode("utf-8")
        with patch("builtins.open", mock_open(read_data=large_data)):
            image = ImageContent.from_path("large_image.jpg")
        assert image.source_type == ImageSourceType.BASE64
        assert image.data == expected_base64
        assert len(image.data) > 1024 * 1024  # Base64 encoding makes it larger


class TestParameterClass:
    """Tests for the Parameter class."""

    def test_parameter_with_type_object(self):
        """Test creating Parameter with actual type objects."""
        param = Parameter(
            name="test_param",
            param_type=str,
            required=True,
            description="A test parameter",
        )
        assert param.name == "test_param"
        assert param.param_type is str
        assert param.required is True
        assert param.description == "A test parameter"

    def test_parameter_with_string_type(self):
        """Test creating Parameter with string type representation."""
        param = Parameter(
            name="test_param",
            param_type="str",
            required=True,
            description="A test parameter",
        )
        assert param.name == "test_param"
        assert param.param_type is str
        assert param.required is True
        assert param.description == "A test parameter"

    @pytest.mark.parametrize(('type_str', 'expected_type'), [
        ('str', str),
        ('int', int),
        ('bool', bool),
        ('float', float),
        ('list', list),
        ('dict', dict),
    ])
    def test_parameter_various_string_types(self, type_str: str, expected_type: type):
        """Test Parameter with various string type representations."""
        param = Parameter(
            name="test_param",
            param_type=type_str,
            required=False,
        )
        assert param.param_type is expected_type

    def test_parameter_with_invalid_string_type(self):
        """Test that invalid string types raise ValueError."""
        with pytest.raises(ValueError, match="Invalid type string"):
            Parameter(
                name="test_param",
                param_type="invalid_type",
                required=True,
            )

    def test_parameter_with_any_type_rejected(self):
        """Test that Any type is rejected."""
        from typing import Any
        with pytest.raises(ValueError, match="Any is not supported"):
            Parameter(
                name="test_param",
                param_type=Any,
                required=True,
            )

    def test_parameter_with_any_type_string_rejected(self):
        """Test that 'Any' string is rejected."""
        with pytest.raises(ValueError, match="Invalid type string"):
            Parameter(
                name="test_param",
                param_type="Any",
                required=True,
            )

    def test_parameter_with_valid_values(self):
        """Test Parameter with valid_values constraint."""
        param = Parameter(
            name="color",
            param_type="str",
            required=True,
            valid_values=["red", "green", "blue"],
        )
        assert param.param_type is str
        assert param.valid_values == ["red", "green", "blue"]

    def test_parameter_with_any_of(self):
        """Test Parameter with any_of for union types."""
        param = Parameter(
            name="id",
            param_type="str",
            required=True,
            any_of=[str, int],
        )
        assert param.param_type is str
        assert param.any_of == [str, int]

    def test_parameter_serialization(self):
        """Test that Parameter can be serialized properly."""
        param = Parameter(
            name="test_param",
            param_type="str",
            required=True,
            description="A test parameter",
        )

        # Test model_dump
        data = param.model_dump()
        assert data['name'] == "test_param"
        assert data['param_type'] == "<class 'str'>"  # Type converted to string representation
        assert data['required'] is True
        assert data['description'] == "A test parameter"

    def test_parameter_with_typing_generic(self):
        """Test Parameter with typing generics like list[str]."""
        param = Parameter(
            name="tags",
            param_type=list[str],
            required=False,
            description="List of tags",
        )
        assert param.param_type == list[str]
        assert param.required is False
