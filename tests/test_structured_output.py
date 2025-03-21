"""Test the Structured Output Wrappers."""
import os
import pytest
from pydantic import BaseModel

from sik_llms import (
    create_client,
    system_message,
    user_message,
    StructuredOutputResponse,
)
from tests.conftest import ANTHROPIC_TEST_MODEL, OPENAI_TEST_MODEL
from dotenv import load_dotenv
load_dotenv()


@pytest.mark.asyncio
@pytest.mark.integration  # these tests make API calls
class TestStructuredOutputs:
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
    @pytest.mark.parametrize('run_async', [True, False])
    async def test__anthropic__structured_outputs(self, model_name: str, run_async: bool):
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
        if run_async:
            response = await client.run_async(messages=messages)
        else:
            response = client(messages=messages)
        assert isinstance(response, StructuredOutputResponse)
        assert isinstance(response.parsed, CalendarEvent)
        assert response.parsed.name
        assert response.parsed.date
        assert response.parsed.participants
        assert 'Alice' in response.parsed.participants
        assert 'Bob' in response.parsed.participants

    @pytest.mark.stochastic(samples=5, threshold=0.5)
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
        assert isinstance(response, StructuredOutputResponse)
        assert isinstance(response.parsed, Contact)
        assert response.parsed.first_name == 'Shane'
        assert response.parsed.last_name == 'Kercheval'
        assert not response.parsed.phone
        assert not response.parsed.email
        assert response.parsed.address.street == '123 Main Street'
        assert not response.parsed.address.street_2
        assert response.parsed.address.city == 'Anytown'
        assert response.parsed.address.state in ('Washington', 'WA')
        assert response.parsed.address.zip_code == '12345'
