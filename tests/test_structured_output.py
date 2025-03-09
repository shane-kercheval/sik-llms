"""Test the Structured Output Wrappers."""
import os
import pytest
from pydantic import BaseModel

from sik_llms import (
    create_client,
    system_message,
    user_message,
    ResponseSummary,
    StructuredOutputResponse,
)
from tests.conftest import ANTHROPIC_TEST_MODEL, OPENAI_TEST_MODEL


@pytest.mark.asyncio
class TestAnthropicStructuredOutputs:
    """Test the OpenAI Structured Output Wrapper."""

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
    async def test__anthropic__structured_outputs(self, model_name: str):
        class CalendarEvent(BaseModel):
            name: str
            date: str
            participants: list[str]

        client = create_client(
            model_name=model_name,
            response_format=CalendarEvent,
        )
        messages=[
            system_message("Extract the event information."),
            user_message("Alice and Bob are going to a science fair on Friday."),
        ]

        response = client(messages=messages)
        assert isinstance(response, ResponseSummary)
        assert isinstance(response.response, StructuredOutputResponse)
        assert isinstance(response.response.parsed, CalendarEvent)
        assert response.response.parsed.name
        assert response.response.parsed.date
        assert response.response.parsed.participants
        assert 'Alice' in response.response.parsed.participants
        assert 'Bob' in response.response.parsed.participants

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
    async def test__anthropic__structured_outputs__nested(self, model_name: str):
        class Address(BaseModel):
            street: str
            street_2: str | None = None
            city: str
            state: str
            zip_code: str

        class Contact(BaseModel):
            first_name: str
            last_name: str
            phone: str | None = None
            email: str | None = None
            address: Address

        client = create_client(
            model_name=model_name,
            response_format=Contact,
        )
        messages=[
            system_message("Extract the information."),
            user_message("Hey my name is Shane Kercheval. I live at 123 Main Street in Anytown, Washington in the USA. The zip code is 12345."),  # noqa: E501
        ]

        response = client(messages=messages)
        assert isinstance(response, ResponseSummary)
        assert isinstance(response.response, StructuredOutputResponse)
        assert isinstance(response.response.parsed, Contact)
        assert response.response.parsed.first_name == 'Shane'
        assert response.response.parsed.last_name == 'Kercheval'
        assert not response.response.parsed.phone
        assert not response.response.parsed.email
        assert response.response.parsed.address.street == '123 Main Street'
        assert not response.response.parsed.address.street_2
        assert response.response.parsed.address.city == 'Anytown'
        assert response.response.parsed.address.state in ('Washington', 'WA')
        assert response.response.parsed.address.zip_code == '12345'
