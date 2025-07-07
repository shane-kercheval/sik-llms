"""Demo script showing sik-llms telemetry capabilities."""
import asyncio
import os
from sik_llms import create_client
import sys


def demonstrate_setup_patterns() -> None:
    """Show both zero-config and manual setup patterns."""
    print("\n" + "="*60)
    print("📋 TELEMETRY SETUP PATTERNS")
    print("="*60)

    print("\n🚀 Pattern 1: Zero-Config (What we're using now)")
    print("   1. pip install sik-llms[telemetry]")
    print("   2. export OTEL_SDK_DISABLED=false")
    print("   3. That's it! sik-llms handles the rest")

    print("\n🏗️  Pattern 2: Manual Setup (Production recommended)")
    print("   1. Set up OpenTelemetry in your app startup code")
    print("   2. export OTEL_SDK_DISABLED=false")
    print("   3. sik-llms detects and respects your configuration")

    print("\n📝 Example manual setup code:")
    print("   ```python")
    print("   from opentelemetry import trace")
    print("   from opentelemetry.sdk.trace import TracerProvider")
    print("   from opentelemetry.sdk.trace.export import BatchSpanProcessor")
    print("   from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter")
    print("   ")
    print("   # Configure your telemetry")
    print("   trace.set_tracer_provider(TracerProvider())")
    print("   tracer_provider = trace.get_tracer_provider()")
    print("   ")
    print("   # Add your exporter")
    print("   otlp_exporter = OTLPSpanExporter(endpoint='https://your-backend.com/v1/traces')")
    print("   span_processor = BatchSpanProcessor(otlp_exporter)")
    print("   tracer_provider.add_span_processor(span_processor)")
    print("   ")
    print("   # Now use sik-llms normally - it will respect your setup!")
    print("   ```")

    print("\n💡 Current demo is using: Zero-Config")

    from sik_llms.telemetry import is_telemetry_enabled
    from opentelemetry import trace
    from opentelemetry.trace import NoOpTracerProvider

    if is_telemetry_enabled():
        provider = trace.get_tracer_provider()
        if isinstance(provider, NoOpTracerProvider):
            print("   Status: ❌ Telemetry enabled but not configured")
        else:
            print("   Status: ✅ Telemetry configured and working")
            print(f"   Provider: {type(provider).__name__}")
    else:
        print("   Status: ⚪ Telemetry disabled")


async def main() -> None:  # noqa: PLR0912, PLR0915
    """Run telemetry demo with various sik-llms features."""
    demonstrate_setup_patterns()

    print("\n🔍 sik-llms Telemetry Demo")
    print(f"📊 Service: {os.getenv('OTEL_SERVICE_NAME', 'sik-llms-demo')}")
    print(f"📡 Endpoint: {os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'http://localhost:4318')}")
    print("\n" + "="*50)

    # Test basic LLM call
    print("\n1. Basic LLM Request")
    client = create_client("gpt-4o-mini")
    try:
        response = client([
            {"role": "user", "content": "What is the capital of France? Keep it brief."},
        ])
        print(f"Response: {response.response}")
        print(f"Tokens: {response.input_tokens} in, {response.output_tokens} out")
        print(f"Cost: ${response.total_cost:.4f}" if response.total_cost else "Cost: N/A")
    except Exception as e:
        print(f"❌ Basic LLM request failed: {e}")
        print("💡 Make sure you have API keys set (OPENAI_API_KEY)")

    # Test async sampling
    print("\n2. Async Sampling (3 responses)")
    try:
        responses = await client.sample([
            {"role": "user", "content": "Give me a random number between 1-100"},
        ], n=3)

        for i, resp in enumerate(responses, 1):
            print(f"Sample {i}: {resp.response}")
    except Exception as e:
        print(f"❌ Async sampling failed: {e}")

    # Test batch generation
    print("\n3. Batch Generation")
    try:
        batch_messages = [
            [{"role": "user", "content": "Name a color"}],
            [{"role": "user", "content": "Name an animal"}],
            [{"role": "user", "content": "Name a food"}],
        ]

        batch_responses = await client.generate_multiple(batch_messages)
        for i, resp in enumerate(batch_responses, 1):
            print(f"Batch {i}: {resp.response}")
    except Exception as e:
        print(f"❌ Batch generation failed: {e}")

    # Test ReasoningAgent if available
    try:
        print("\n4. ReasoningAgent")
        from sik_llms import ReasoningAgent

        reasoning_client = ReasoningAgent(
            model_name="gpt-4o-mini",
            max_iterations=2,  # Keep demo short
        )

        print("Starting reasoning process...")
        reasoning_response = reasoning_client([
            {"role": "user", "content": "What's 15 * 23? Show your work."},
        ])
        print(f"Reasoning result: {reasoning_response.response}")

    except Exception as e:
        print(f"❌ ReasoningAgent test failed: {e}")

    # Test Anthropic client if available
    print("\n5. Anthropic Client")
    try:
        anthropic_client = create_client("claude-3-5-haiku-latest")
        anthropic_response = anthropic_client([
            {"role": "user", "content": "Say hello in exactly 3 words"},
        ])
        print(f"Anthropic response: {anthropic_response.response}")
        print(f"Tokens: {anthropic_response.input_tokens} in, {anthropic_response.output_tokens} out")  # noqa: E501
    except Exception as e:
        print(f"❌ Anthropic test failed: {e}")
        print("💡 Make sure you have ANTHROPIC_API_KEY set")

    # Test span linking demonstration with new TraceContext
    print("\n6. TraceContext Demo (New Feature!)")
    try:
        # Generate content and get trace context automatically
        print("Generating content with automatic trace context...")
        content_response = client([
            {"role": "user", "content": "Write a haiku about programming"},
        ])

        print(f"Content: {content_response.response}")

        # Check if trace context is available
        if content_response.trace_context:
            print("✅ Trace context captured automatically!")
            print(f"   Trace ID: {content_response.trace_context.trace_id}")
            print(f"   Span ID: {content_response.trace_context.span_id}")

            # Demonstrate span link creation
            evaluation_link = content_response.trace_context.create_link({
                "link.type": "evaluation_of_generation",
                "evaluation.type": "haiku_quality",
                "content.type": "poetry",
            })

            if evaluation_link:
                print("✅ Evaluation link created successfully")
                print("   This link can be used to correlate evaluation results")
                print("   back to the original generation in Jaeger!")
            else:
                print("⚠️  Link creation failed (OpenTelemetry not available)")
        else:
            print("⚪ No trace context available (telemetry may be disabled)")

        # Demonstrate with async call too
        print("\n   Testing with async call...")
        async_response = await client.run_async([
            {"role": "user", "content": "Name a programming language"},
        ])

        if async_response.trace_context:
            print("✅ Async call also has trace context!")
            print(f"   Async Trace ID: {async_response.trace_context.trace_id}")
        else:
            print("⚪ No trace context from async call")

    except Exception as e:
        print(f"❌ TraceContext demo failed: {e}")

    print("\n" + "="*50)
    print("🎉 Demo complete!")
    print("\n📊 View traces at: http://localhost:16686")
    print("🔍 Search for service: sik-llms-demo")
    print("\n💡 Try different queries:")
    print("  - operation_name=llm.request")
    print("  - llm.model=gpt-4o-mini")
    print("  - llm.operation=chat")
    print("  - llm.provider=openai")
    print("\n🎯 Telemetry Features Demonstrated:")
    print("  ✅ Basic LLM request tracing")
    print("  ✅ Async sampling with parallel spans")
    print("  ✅ Batch processing telemetry")
    print("  ✅ ReasoningAgent iteration tracking")
    print("  ✅ Multi-provider support (OpenAI + Anthropic)")
    print("  ✅ TraceContext for easy span linking (NEW!)")
    print("  ✅ Token usage and cost metrics")
    print("\n🆕 TraceContext Benefits:")
    print("  ✅ No manual wrapper spans needed")
    print("  ✅ Automatic trace context extraction")
    print("  ✅ Works with both sync and async calls")
    print("  ✅ Built-in link creation for evaluation")


if __name__ == "__main__":
    # Verify telemetry is enabled
    if os.getenv("OTEL_SDK_DISABLED", "true").lower() != "false":
        print("❌ Telemetry not enabled!")
        print("Run: export OTEL_SDK_DISABLED=false")
        print("Or use: make telemetry-demo")
        sys.exit(1)

    # Check if basic API keys are available
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))

    if not has_openai and not has_anthropic:
        print("⚠️  No API keys detected!")
        print("Set OPENAI_API_KEY or ANTHROPIC_API_KEY for full demo")
        print("Some features will be mocked for demonstration")
    elif has_openai and not has_anthropic:
        print("📝 OpenAI API key detected - OpenAI features available")
        print("💡 Set ANTHROPIC_API_KEY for Anthropic demo features")
    elif has_anthropic and not has_openai:
        print("📝 Anthropic API key detected - Anthropic features available")
        print("💡 Set OPENAI_API_KEY for OpenAI demo features")
    else:
        print("🚀 Both OpenAI and Anthropic API keys detected - full demo available!")

    print("\n⏳ Starting telemetry demo...")
    print("💡 Traces will appear in Jaeger within a few seconds")

    asyncio.run(main())
