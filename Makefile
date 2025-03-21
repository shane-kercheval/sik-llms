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
	uv run ruff check tests

unittests:
	# only runs a subset of tests that are generally faster
	uv run pytest --durations=0 --durations-min=0.1 -k "not integration" tests

all_test:
	uv run pytest --durations=0 --durations-min=0.1 tests

tests: linting all_test

package-build:
	rm -rf dist/*
	uv build --no-sources

package-publish:
	uv publish --token ${UV_PUBLISH_TOKEN}

package: package-build package-publish
