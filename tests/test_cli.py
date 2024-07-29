from subprocess import CompletedProcess, run
import pytest


@pytest.fixture()
def endpoint():
    return "http://localhost:11434"


@pytest.fixture()
def model():
    return "llama3"


@pytest.fixture()
def run_prefix():
    return "python -m ollama"


def test_endpoint_connection(run_prefix, endpoint, model):
    command = " ".join([run_prefix, "-e", endpoint, "-m", model])
    r = run(command)
    assert isinstance(r, CompletedProcess)
    assert r.args == command
    assert r.returncode == 0


def test_hello_world(run_prefix, endpoint, model):
    command = " ".join(
        [run_prefix, "-e", endpoint, "-m", model, "-p", '"Hello world?"']
    )
    r = run(command)
    assert isinstance(r, CompletedProcess)
    assert r.args == command
    assert r.returncode == 0
