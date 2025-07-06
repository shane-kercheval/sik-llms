.PHONY: tests

-include .env
export

####
# Environment
####
build-env:
	uv sync

linting_src:
	uv run ruff check src/sik_llms --fix --unsafe-fixes

linting_examples:
	uv run ruff check examples --fix --unsafe-fixes

linting_tests:
	uv run ruff check tests --fix --unsafe-fixes

linting:
	# Run all linters on source, examples, and tests
	uv run ruff check src/sik_llms examples tests --fix --unsafe-fixes

quicktests:
	# only runs a subset of tests that are generally faster
	uv run pytest --durations=0 --durations-min=0.1 -k "not integration" tests

unittests:
	uv run pytest --durations=0 --durations-min=0.1 tests

# Telemetry-specific testing
telemetry-tests:
	# Run integration tests requiring Jaeger
	uv run pytest -m "integration" tests/test_telemetry.py -v

telemetry-unit-tests:
	# Run only telemetry unit tests (no external dependencies)
	uv run pytest -m "not integration" tests/test_telemetry.py tests/test_telemetry_regression.py -v

# Enhanced test commands
quicktests-with-telemetry:
	# Run quicktests + telemetry unit tests
	uv run pytest --durations=0 --durations-min=0.1 -k "not integration" tests

telemetry-all:
	uv run pytest --durations=0 --durations-min=0.1 tests/test_telemetry.py tests/test_telemetry_regression.py -v

tests: linting unittests telemetry-all

package-build:
	rm -rf dist/*
	uv build --no-sources

package-publish:
	uv publish --token ${UV_PUBLISH_TOKEN}

package: package-build package-publish

# example MCP client & chat
chat:
	uv run python examples/cli.py \
		-chat \
		--mcp_config examples/mcp_fake_server_config.json \
		--model 'gpt-4o'
# --model claude-3-5-sonnet-latest

tools:
	uv run python examples/cli.py \
		--mcp_config examples/mcp_fake_server_config.json \
		-tools

####
# Telemetry Development
####

# Jaeger management for local development
jaeger-start:
	@echo "Starting Jaeger for telemetry testing..."
	docker run -d --name jaeger-sik-llms \
		-p 16686:16686 \
		-p 4318:4318 \
		jaegertracing/all-in-one:latest
	@echo "Jaeger UI available at http://localhost:16686"

jaeger-stop:
	@echo "Stopping Jaeger..."
	docker stop jaeger-sik-llms || true
	docker rm jaeger-sik-llms || true

jaeger-logs:
	@echo "Jaeger container logs:"
	docker logs jaeger-sik-llms

# Telemetry example/demo
telemetry-demo:
	@echo "Running telemetry demo (requires Jaeger)..."
	OTEL_SDK_DISABLED=false \
	OTEL_SERVICE_NAME="sik-llms-demo" \
	OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318" \
	uv run python examples/telemetry_demo.py
