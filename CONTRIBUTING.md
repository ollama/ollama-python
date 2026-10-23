# Contributing to Ollama Python

Thank you for your interest in contributing to the official Ollama Python library! We welcome bug reports, bug fixes, feature proposals, documentation improvements, and example additions.

---

## Prerequisites

- **Python 3.8+** (Python 3.10+ recommended for development)
- **[uv](https://docs.astral.sh/uv/)** or **[hatch](https://hatch.pypa.io/)**
- A running instance of **[Ollama](https://ollama.com/download)** (optional for running unit/mock tests, required for real model execution)

---

## Development Setup

1. **Fork and clone the repository:**

   ```bash
   git clone https://github.com/<your-username>/ollama-python.git
   cd ollama-python
   ```

2. **Set up virtual environment and install dependencies:**

   Using `uv`:
   ```bash
   uv sync --all-extras --all-groups
   ```

   Using `hatch`:
   ```bash
   hatch env create
   ```

---

## Code Style and Quality

We use [Ruff](https://docs.astral.sh/ruff/) to format and lint our codebase.

- **Indentation**: 2 spaces (`indent-width = 2`, `indent-style = 'space'`)
- **Quotes**: Single quotes (`quote-style = 'single'`)
- **Line length**: 320 characters max

### Formatting & Linting Commands

Check formatting and linting:
```bash
uv run ruff format --check .
uv run ruff check .
```

Automatically fix format and lint issues:
```bash
uv run ruff format .
uv run ruff check --fix .
```

Or using `hatch`:
```bash
hatch fmt --check
```

---

## Running Tests

All unit tests use mock HTTP servers with `pytest-httpserver` and doctests across module docstrings.

Run the test suite with `uv`:
```bash
uv run pytest
```

Run tests across environments with `hatch`:
```bash
hatch test
```

Run a specific test file:
```bash
uv run pytest tests/test_client.py
uv run pytest tests/test_utils.py
uv run pytest tests/test_type_serialization.py
```

---

## Updating Lockfile and Requirements

If dependencies in `pyproject.toml` are modified, make sure the lockfile and exported requirements remain in sync:

```bash
uv lock
uv export > requirements.txt
```

Verify that requirements match:
```bash
git diff --exit-code requirements.txt
```

---

## Pull Request Guidelines

1. **Branch naming**: Use descriptive branch names (e.g. `fix/named-tempfile-windows`, `feat/client-api-key-arg`).
2. **Commit messages**: Follow concise conventional commit style:
   - `fix: ...` for bug fixes
   - `feat: ...` for new features
   - `docs: ...` for documentation updates
   - `test: ...` for test additions
   - `refactor: ...` for code refactoring
3. **Tests**: Add unit tests for any new functionality or bug fixes in `tests/`.
4. **Documentation & Examples**: Update `README.md` and `examples/README.md` if changing public APIs or adding new features.
5. **No regressions**: Ensure formatting (`ruff format`), linting (`ruff check`), and all tests pass before submitting.
