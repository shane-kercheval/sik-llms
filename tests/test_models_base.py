"""Tests for the models_base module."""
from sik_llms import Client, RegisteredClients, system_message
from sik_llms.models_base import assistant_message, user_message

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
        )
        assert isinstance(client, RegisteredMockClient)
        assert client.model_name == 'test-model'
        assert client.kwargs['temperature'] == 0.7

        # Test instantiation with enum registration
        client = Client.instantiate(
            RegisteredClients.OPENAI,
            model_name='gpt-4o',
            max_tokens=100,
        )
        assert client.model == 'gpt-4o'
        assert client.model_parameters['max_tokens'] == 100

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
