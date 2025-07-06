"""Demo script showing sik-llms telemetry capabilities."""
import asyncio
import os
from sik_llms import create_client


async def main():
    """Run telemetry demo with various sik-llms features."""
    print("🔍 sik-llms Telemetry Demo")
    print(f"📊 Service: {os.getenv('OTEL_SERVICE_NAME', 'sik-llms-demo')}")
    print(f"📡 Endpoint: {os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'http://localhost:4318')}")
    print("\n" + "="*50)
    
    # Test basic LLM call
    print("\n1. Basic LLM Request")
    client = create_client("gpt-4o-mini")
    try:
        response = client([
            {"role": "user", "content": "What is the capital of France? Keep it brief."}
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
            {"role": "user", "content": "Give me a random number between 1-100"}
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
            max_iterations=2  # Keep demo short
        )
        
        print("Starting reasoning process...")
        reasoning_response = reasoning_client([
            {"role": "user", "content": "What's 15 * 23? Show your work."}
        ])
        print(f"Reasoning result: {reasoning_response.response}")
        
    except Exception as e:
        print(f"❌ ReasoningAgent test failed: {e}")
    
    # Test Anthropic client if available
    print("\n5. Anthropic Client")
    try:
        anthropic_client = create_client("claude-3-5-haiku-latest")
        anthropic_response = anthropic_client([
            {"role": "user", "content": "Say hello in exactly 3 words"}
        ])
        print(f"Anthropic response: {anthropic_response.response}")
        print(f"Tokens: {anthropic_response.input_tokens} in, {anthropic_response.output_tokens} out")
    except Exception as e:
        print(f"❌ Anthropic test failed: {e}")
        print("💡 Make sure you have ANTHROPIC_API_KEY set")
    
    # Test span linking demonstration
    print("\n6. Span Linking Demo")
    try:
        from sik_llms import create_span_link
        
        # Create a mock span link (in real use, you'd get these from actual spans)
        mock_trace_id = "1234567890abcdef1234567890abcdef"
        mock_span_id = "fedcba0987654321"
        
        link = create_span_link(
            trace_id=mock_trace_id,
            span_id=mock_span_id,
            attributes={"link.type": "evaluation_of_generation"}
        )
        
        if link:
            print("✅ Span link created successfully")
        else:
            print("⚪ Span linking not available (telemetry disabled)")
            
    except Exception as e:
        print(f"❌ Span linking demo failed: {e}")
    
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
    print("  ✅ Span linking for evaluation correlation")
    print("  ✅ Token usage and cost metrics")


if __name__ == "__main__":
    # Verify telemetry is enabled
    if os.getenv("OTEL_SDK_DISABLED", "true").lower() != "false":
        print("❌ Telemetry not enabled!")
        print("Run: export OTEL_SDK_DISABLED=false")
        print("Or use: make telemetry-demo")
        exit(1)
    
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