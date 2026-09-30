import pytest

def parse_model_tag(model: str) -> tuple[str, str]:
    cleaned = model.strip()
    if ":" in cleaned:
        parts = cleaned.split(":", 1)
        return parts[0].strip(), parts[1].strip()
    return cleaned, "latest"

def test_parse_model_tag():
    assert parse_model_tag("llama3") == ("llama3", "latest")
    assert parse_model_tag("mistral:7b") == ("mistral", "7b")
    assert parse_model_tag("  qwen2.5:14b  ") == ("qwen2.5", "14b")
