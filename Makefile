.PHONY: install check lint fmt test clean

# uv 默认写 ~/.cache/uv，本机沙箱不允许；放进项目内也避免污染全局
export UV_CACHE_DIR ?= $(CURDIR)/.uv-cache
export UV_PYTHON_INSTALL_DIR ?= $(CURDIR)/.uv-python

install:
	uv python install 3.12
	uv sync

check: lint test

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
