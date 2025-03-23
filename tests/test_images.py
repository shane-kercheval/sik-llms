"""Tests for the image content functionality for clients."""
import base64
import os
from pathlib import Path
import pytest
from sik_llms import (
    create_client,
    user_message,
    ImageContent,
    TextResponse,
)
from tests.conftest import ANTHROPIC_TEST_MODEL, OPENAI_TEST_MODEL


@pytest.fixture
def test_image_bytes() -> bytes:
    """Create a small test image in bytes."""
    base_str = "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"  # noqa: E501
    return base64.b64decode(base_str.encode('utf-8'))


@pytest.mark.asyncio
@pytest.mark.integration  # these tests make API calls
@pytest.mark.parametrize('model_name', [
    pytest.param(
        OPENAI_TEST_MODEL,
        id="OpenAI",
    ),
    pytest.param(
        ANTHROPIC_TEST_MODEL,
        id="Anthropic",
        marks=pytest.mark.skipif(
            os.getenv('ANTHROPIC_API_KEY') is None,
            reason="ANTHROPIC_API_KEY is not set",
        ),
    ),
])
class TestImagesChat:
    """Test image handling across different providers."""

    async def test_url_image(self, model_name: str):
        """Test handling of URL-based images."""
        image = ImageContent.from_url(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/320px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
        )

        client = create_client(model_name=model_name)
        response = await client.run_async(messages=[
            user_message([
                "What's in this image? Describe it briefly.",
                image,
            ]),
        ])
        assert isinstance(response, TextResponse)
        assert response.response.strip()  # Response should not be empty
        assert response.input_tokens > 0
        assert response.output_tokens > 0
        assert response.input_cost > 0
        assert response.output_cost > 0

    async def test_local_file_image(self, model_name: str, test_image_bytes: bytes, tmp_path: Path):  # noqa: E501
        """Test handling of local file images."""
        # Create temporary file with test image bytes
        test_file = tmp_path / "test_image.jpg"
        test_file.write_bytes(test_image_bytes)
        image = ImageContent.from_path(test_file)
        client = create_client(model_name=model_name)
        response = await client.run_async(messages=[
            user_message([
                "What is this image showing? Describe it briefly.",
                image,
            ]),
        ])
        assert isinstance(response, TextResponse)
        assert 'basketball' in response.response.strip().lower()  # Response should not be empty
        assert response.input_tokens > 0
        assert response.output_tokens > 0
        assert response.input_cost > 0
        assert response.output_cost > 0

    async def test_bytes_image(self, model_name: str, test_image_bytes: bytes):
        """Test handling of image bytes."""
        image = ImageContent.from_bytes(
            test_image_bytes,
            media_type="image/jpeg",
        )
        client = create_client(model_name=model_name)
        response = await client.run_async(messages=[
            user_message([
                "What is this image showing? Describe it briefly.",
                image,
            ]),
        ])
        assert isinstance(response, TextResponse)
        assert 'basketball' in response.response.strip().lower()  # Response should not be empty
        assert response.input_tokens > 0
        assert response.output_tokens > 0
        assert response.input_cost > 0
        assert response.output_cost > 0

    async def test_multiple_images(self, model_name: str, test_image_bytes: Path):
        """Test handling multiple images in one message."""
        image1 = ImageContent.from_url(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/320px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
        )
        image2 = ImageContent.from_bytes(
            test_image_bytes,
            media_type="image/jpeg",
        )

        client = create_client(model_name=model_name)
        response = await client.run_async(messages=[
            user_message([
                "I'll show you two images.",
                image1,
                "This is the first image above. Here's the second:",
                image2,
                "What is the second image showing? Describe it briefly.",
            ]),
        ])

        assert isinstance(response, TextResponse)
        assert 'basketball' in response.response.lower()
        assert response.input_tokens > 0
        assert response.output_tokens > 0
        assert response.input_cost > 0
        assert response.output_cost > 0

    async def test_image_in_conversation(self, model_name: str, test_image_bytes: bytes):
        """Test handling images within a conversation context."""
        image = ImageContent.from_bytes(
            test_image_bytes,
            media_type="image/jpeg",
        )
        client = create_client(model_name=model_name)
        response = await client.run_async(messages=[
            user_message([
                "Here's the image:",
                image,
            ]),
            user_message("What is this image showing? Describe it briefly."),
        ])
        assert isinstance(response, TextResponse)
        assert 'basketball' in response.response.strip().lower()
        assert response.input_tokens > 0
        assert response.output_tokens > 0
        assert response.input_cost > 0
        assert response.output_cost > 0
