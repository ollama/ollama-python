
.PHONY: lint format test add-dev

# write help 
help:
	@echo "Available commands:"
	@echo "  lint      - Run linters"
	@echo "  format    - Format code"
	@echo "  test      - Run tests"
	@echo "  add-dev   - Add development dependencies"

# Use uv to run tools in the project's environment. Example:
#   uv add --dev ruff

lint:
	uv ruff check .

format:
	uv ruff format .

add-dev:
	# Add ruff to the project's dev tools using uv
	uv add --dev ruff

test:
	python -m pytest -q
