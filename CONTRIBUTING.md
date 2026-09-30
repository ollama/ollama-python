# Contributing to ollama-python

Thanks for your interest in contributing! This guide covers the local development workflow used by this repository.

## Prerequisites

- Python 3.8+
- [uv](https://docs.astral.sh/uv/) (recommended) or another way to run Hatch environments
- [Ollama](https://ollama.com/download) installed if you want to exercise live examples against a local server

## Setup

Clone the repository and install the package in editable mode with development tools via Hatch:

```sh
git clone https://github.com/ollama/ollama-python.git
cd ollama-python
uvx hatch env create
```

Alternatively, with pip:

```sh
pip install -e .
pip install pytest pytest-anyio pytest-httpserver ruff
```

## Repository layout

- `ollama/` — library source (`_client.py`, `_types.py`, `_utils.py`)
- `tests/` — unit tests (HTTP server fixtures; no live Ollama required for CI)
- `examples/` — usage examples
- `pyproject.toml` — project metadata, Ruff, and Hatch configuration

## Tests

Run the full test suite the same way CI does:

```sh
uvx hatch test -acp
```

Or with pytest directly after installing test dependencies:

```sh
pytest ollama tests
```

## Linting and formatting

This project uses [Ruff](https://docs.astral.sh/ruff/) with settings from `pyproject.toml` (2-space indent, single quotes).

Check formatting and lint (CI commands):

```sh
uvx hatch fmt --check -f
uvx hatch fmt --check -l
```

Apply fixes locally:

```sh
uvx hatch fmt
```

Also keep lockfiles in sync when changing dependencies:

```sh
uv lock --check
uv export > requirements.txt
```

## Pull requests

- Create a branch from `main` and keep the change focused
- Ensure tests and Ruff checks pass before opening a PR
- Prefer clear commit messages that explain *why*
- Link related issues in the PR description (for example `Fixes #123`)
- Draft PRs are welcome for early feedback

## Questions

Open an issue at https://github.com/ollama/ollama-python/issues if something in this guide is unclear or outdated.
