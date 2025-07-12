#!/usr/bin/env python3
"""Simple telemetry test for make command."""

from sik_llms import create_client

def main() -> None:  # noqa: D103
    print('Creating client...')
    client = create_client('gpt-4o-mini')

    print('Sending test request...')
    response = client([{'role': 'user', 'content': 'Say hello for telemetry test'}])

    print(f'Response: {response.response[:50]}...')

    if response.trace_context:
        print(f'Trace ID: {response.trace_context.trace_id}')
    else:
        print('Trace ID: None')

    print('Test complete! Check Jaeger UI for traces.')

if __name__ == "__main__":
    main()
