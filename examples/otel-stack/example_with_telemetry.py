#!/usr/bin/env python3
"""
Example script demonstrating sik-llms OpenTelemetry integration.

This script shows how to use sik-llms with the complete observability stack
from examples/otel_101.md. It demonstrates:

1. Automatic trace and metric generation
2. TraceContext capture for span linking
3. ReasoningAgent instrumentation
4. Custom evaluation with trace linking

Prerequisites:
- pip install sik-llms[telemetry]
- Valid OpenAI API key in environment
- OpenTelemetry stack running (make start)

Usage:
    # Start the observability stack first
    make start

    # Enable telemetry
    export OTEL_SDK_DISABLED=false
    export OTEL_SERVICE_NAME="sik-llms-example"
    export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318"

    # Run the example
    python example_with_telemetry.py

    # View results at:
    # - Traces: http://localhost:16686 (Jaeger)
    # - Metrics: http://localhost:9090 (Prometheus)
    # - Dashboards: http://localhost:3000 (Grafana admin/admin)
"""

import os
import time
from opentelemetry import trace
import sys
from sik_llms import create_client
from sik_llms.reasoning_agent import ReasoningAgent


def check_telemetry_setup() -> bool:
    """Verify telemetry is properly configured."""
    print("🔧 Checking telemetry setup...")

    otel_disabled = os.getenv("OTEL_SDK_DISABLED", "").lower()
    service_name = os.getenv("OTEL_SERVICE_NAME", "")
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")

    if otel_disabled == "true":
        print("❌ OTEL_SDK_DISABLED=true - telemetry is disabled")
        return False

    if not service_name:
        print("⚠️  OTEL_SERVICE_NAME not set - will use default")

    if not endpoint:
        print("⚠️  OTEL_EXPORTER_OTLP_ENDPOINT not set - will use default")

    print(f"✅ Service: {service_name or 'default'}")
    print(f"✅ Endpoint: {endpoint or 'http://localhost:4318'}")
    return True


def basic_llm_example():  # noqa: ANN201
    """Basic LLM call with automatic telemetry."""
    print("\n📝 Running basic LLM example...")

    # This automatically creates traces AND metrics
    client = create_client("gpt-4o-mini")
    response = client([{
        "role": "user",
        "content": "Explain what OpenTelemetry is in one sentence.",
    }])

    print(f"Response: {response.response}")

    # Check trace context (NEW feature!)
    if response.trace_context:
        print(f"🔗 Trace ID: {response.trace_context.trace_id}")
        print(f"🔗 Span ID: {response.trace_context.span_id}")
        print("✅ TraceContext captured - ready for span linking!")
    else:
        print("⚪ No trace context (telemetry may be disabled)")

    return response


def reasoning_agent_example():  # noqa: ANN201
    """ReasoningAgent with detailed instrumentation."""
    print("\n🧠 Running ReasoningAgent example...")

    # ReasoningAgent creates rich trace trees automatically
    agent = ReasoningAgent(model_name="gpt-4o-mini")
    response = agent([{
        "role": "user",
        "content": "Calculate 15 * 23 + 7 and explain your reasoning step by step.",
    }])

    print(f"Response: {response.response[:200]}...")

    if response.trace_context:
        print(f"🔗 Reasoning Trace ID: {response.trace_context.trace_id}")
        print("✅ Detailed reasoning spans available in Jaeger!")

    return response


def evaluation_with_linking_example(generation_response):  # noqa: ANN001, ANN201
    """Example of evaluation with trace linking using TraceContext."""
    print("\n📊 Running evaluation with trace linking...")

    if not generation_response.trace_context:
        print("⚠️  No trace context available - skipping linking example")
        return None

    tracer = trace.get_tracer("evaluation_system")

    # Create link back to the original generation
    evaluation_link = generation_response.trace_context.create_link({
        "link.type": "quality_evaluation",
        "evaluation.category": "helpfulness",
        "evaluation.framework": "example_script",
    })

    # Run evaluation with the link
    links = [evaluation_link] if evaluation_link else []
    with tracer.start_as_current_span("content_evaluation", links=links) as eval_span:
        # Simulate evaluation
        time.sleep(0.1)  # Simulate processing time

        # Mock evaluation scores
        clarity_score = 8.5
        accuracy_score = 9.2
        helpfulness_score = 8.8
        overall_score = (clarity_score + accuracy_score + helpfulness_score) / 3

        # Record evaluation results as span attributes
        eval_span.set_attribute("evaluation.clarity_score", clarity_score)
        eval_span.set_attribute("evaluation.accuracy_score", accuracy_score)
        eval_span.set_attribute("evaluation.helpfulness_score", helpfulness_score)
        eval_span.set_attribute("evaluation.overall_score", overall_score)
        eval_span.set_attribute("evaluation.pass", overall_score >= 8.0)

        print("📊 Evaluation scores:")
        print(f"   Clarity: {clarity_score}")
        print(f"   Accuracy: {accuracy_score}")
        print(f"   Helpfulness: {helpfulness_score}")
        print(f"   Overall: {overall_score:.1f}")
        print(f"✅ Evaluation complete - linked to trace {generation_response.trace_context.trace_id}")  # noqa: E501

        return {
            "clarity": clarity_score,
            "accuracy": accuracy_score,
            "helpfulness": helpfulness_score,
            "overall": overall_score,
        }


def multi_stage_pipeline_example():  # noqa: ANN201
    """Multi-stage pipeline with automatic trace linking."""
    print("\n🔄 Running multi-stage pipeline example...")

    tracer = trace.get_tracer("pipeline_system")
    client = create_client("gpt-4o-mini")

    with tracer.start_as_current_span("content_pipeline") as pipeline_span:
        # Stage 1: Create outline
        print("  Stage 1: Creating outline...")
        outline_response = client([{
            "role": "user",
            "content": "Create a brief outline for explaining machine learning to beginners.",
        }])

        # Stage 2: Expand outline
        print("  Stage 2: Expanding outline...")
        expanded_response = client([{
            "role": "user",
            "content": f"Expand this outline into a detailed explanation:\n{outline_response.response}",  # noqa: E501
        }])

        # Record pipeline metadata
        pipeline_span.set_attribute("pipeline.stages", 2)
        pipeline_span.set_attribute("pipeline.type", "content_generation")

        if outline_response.trace_context:
            pipeline_span.set_attribute(
                "pipeline.stage1.trace_id",
                outline_response.trace_context.trace_id,
            )
        if expanded_response.trace_context:
            pipeline_span.set_attribute(
                "pipeline.stage2.trace_id",
                expanded_response.trace_context.trace_id,
            )

        print(f"📄 Final content: {expanded_response.response[:150]}...")
        print("✅ Pipeline complete - check Jaeger for full trace tree!")

        return expanded_response


def demonstrate_metrics() -> None:
    """Show what metrics are automatically generated."""
    print("\n📈 Demonstrating automatic metrics...")
    print("sik-llms automatically emits these metrics to Prometheus:")
    print("  • llm_tokens_input_total - Total input tokens")
    print("  • llm_tokens_output_total - Total output tokens")
    print("  • llm_cost_total_usd - Total cost in USD")
    print("  • llm_request_duration_seconds - Request latency")
    print("  • llm_reasoning_iterations_total - Reasoning iterations")
    print("  • llm_reasoning_tool_calls_total - Tool calls")
    print()
    print("💡 View these metrics at:")
    print("  📊 Prometheus: http://localhost:9090")
    print("  📋 Grafana: http://localhost:3000 (admin/admin)")
    print()
    print("🔍 Example Prometheus queries:")
    print("  rate(llm_tokens_input_total[5m])  # Tokens per minute")
    print("  llm_request_duration_seconds      # Request latency")
    print("  rate(llm_cost_total_usd[1h])      # Cost per hour")


def main() -> None:
    """Run all examples demonstrating OpenTelemetry integration."""
    print("🎯 sik-llms OpenTelemetry Integration Example")
    print("=" * 50)

    # Check setup
    if not check_telemetry_setup():
        print("\n❌ Please fix telemetry configuration and try again.")
        print("💡 Run: make start  # to start the observability stack")
        sys.exit(1)

    try:
        # Run examples
        print("\n🚀 Running telemetry examples...")

        # Basic example with TraceContext
        basic_response = basic_llm_example()

        # ReasoningAgent example
        reasoning_agent_example()

        # Evaluation with linking
        evaluation_with_linking_example(basic_response)

        # Multi-stage pipeline
        multi_stage_pipeline_example()

        # Show metrics info
        demonstrate_metrics()

        print("\n✅ All examples completed!")
        print("\n🔍 View your telemetry data:")
        print("  📊 Traces: http://localhost:16686 (Jaeger)")
        print("  📈 Metrics: http://localhost:9090 (Prometheus)")
        print("  📋 Dashboards: http://localhost:3000 (Grafana admin/admin)")
        print("\n💡 Try searching for:")
        print(f"  Service: {os.getenv('OTEL_SERVICE_NAME', 'sik-llms')}")
        print("  Operations: llm.request, llm.reasoning_agent.stream")
        print("  Tags: llm.model=gpt-4o-mini")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("\n🔧 Troubleshooting:")
        print("  1. Ensure OpenAI API key is set: export OPENAI_API_KEY=...")
        print("  2. Start observability stack: make start")
        print("  3. Enable telemetry: export OTEL_SDK_DISABLED=false")
        print("  4. Install dependencies: pip install sik-llms[telemetry]")


if __name__ == "__main__":
    main()
