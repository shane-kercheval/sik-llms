.PHONY: tests

-include .env
export

####
# Environment
####
build-env:
	uv sync

linting:
	uv run ruff check src/sik_llms
	uv run ruff check examples/cli.py
	uv run ruff check tests

quicktests:
	# only runs a subset of tests that are generally faster
	uv run pytest --durations=0 --durations-min=0.1 -k "not integration" tests

unittests:
	uv run pytest --durations=0 --durations-min=0.1 tests

tests: linting unittests

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
# --model claude-3-5-sonnet

tools:
	uv run python examples/cli.py \
		--mcp_config examples/mcp_fake_server_config.json \
		-tools
