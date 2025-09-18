import base64
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from ollama._auth import OllamaAuth


@pytest.fixture
def temp_key_pair():
    """Create a temporary Ed25519 key pair for testing."""
    # Generate a test key pair
    private_key = ed25519.Ed25519PrivateKey.generate()
    
    # Serialize the private key in OpenSSH format
    private_key_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.OpenSSH,
        encryption_algorithm=serialization.NoEncryption()
    )
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='_ed25519') as f:
        f.write(private_key_bytes)
        temp_key_path = f.name
    
    yield temp_key_path, private_key
    
    # Cleanup
    try:
        os.unlink(temp_key_path)
    except FileNotFoundError:
        pass


@pytest.fixture
def invalid_key_file():
    """Create a temporary file with invalid key content."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='_invalid') as f:
        f.write("This is not a valid private key")
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    try:
        os.unlink(temp_path)
    except FileNotFoundError:
        pass


class TestOllamaAuth:
    """Test suite for OllamaAuth class."""

    def test_init_default_key_path(self):
        """Test initialization with default key path."""
        auth = OllamaAuth()
        expected_path = os.path.join(str(Path.home()), '.ollama', 'id_ed25519')
        assert auth.key_path == expected_path

    def test_init_custom_key_path(self):
        """Test initialization with custom key path."""
        custom_path = "/custom/path/to/key"
        auth = OllamaAuth(key_path=custom_path)
        assert auth.key_path == custom_path

    def test_init_expanduser_path(self):
        """Test initialization with path containing ~ expansion."""
        auth = OllamaAuth(key_path="~/custom/key")
        expected_path = os.path.expanduser("~/custom/key")
        assert auth.key_path == expected_path

    def test_init_expandvars_path(self):
        """Test initialization with path containing environment variables."""
        with patch.dict(os.environ, {'TEST_DIR': '/test/dir'}):
            auth = OllamaAuth(key_path="$TEST_DIR/key")
            assert auth.key_path == "/test/dir/key"

    def test_load_private_key_success(self, temp_key_pair):
        """Test successful private key loading."""
        temp_key_path, expected_private_key = temp_key_pair
        
        auth = OllamaAuth(key_path=temp_key_path)
        loaded_key = auth.load_private_key()
        
        # Verify the loaded key is an Ed25519 private key
        assert isinstance(loaded_key, ed25519.Ed25519PrivateKey)
        
        # Verify the public keys match (indirect way to verify it's the same key)
        expected_public = expected_private_key.public_key()
        loaded_public = loaded_key.public_key()
        
        expected_bytes = expected_public.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )
        loaded_bytes = loaded_public.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )
        
        assert expected_bytes == loaded_bytes

    def test_load_private_key_file_not_found(self):
        """Test FileNotFoundError when key file doesn't exist."""
        auth = OllamaAuth(key_path="/nonexistent/path/key")
        
        with pytest.raises(FileNotFoundError, match="Could not find Ollama private key"):
            auth.load_private_key()

    def test_load_private_key_invalid_key(self, invalid_key_file):
        """Test ValueError when key file contains invalid data."""
        auth = OllamaAuth(key_path=invalid_key_file)
        
        with pytest.raises(ValueError, match="Invalid private key"):
            auth.load_private_key()

    def test_get_public_key_b64(self, temp_key_pair):
        """Test base64 public key extraction."""
        temp_key_path, private_key = temp_key_pair
        
        auth = OllamaAuth(key_path=temp_key_path)
        public_key_b64 = auth.get_public_key_b64(private_key)
        
        # Verify it's a valid base64 string
        try:
            decoded = base64.b64decode(public_key_b64)
            assert len(decoded) > 0
        except Exception:
            pytest.fail("Returned value is not valid base64")
        
        # Verify the format by checking it matches the OpenSSH format
        public_key = private_key.public_key()
        openssh_pub = public_key.public_bytes(
            encoding=serialization.Encoding.OpenSSH,
            format=serialization.PublicFormat.OpenSSH,
        ).decode('utf-8').strip()
        
        expected_b64 = openssh_pub.split(' ')[1]
        assert public_key_b64 == expected_b64

    def test_get_public_key_b64_malformed_key(self):
        """Test ValueError when OpenSSH public key is malformed."""
        auth = OllamaAuth()
        
        # Create a mock private key that produces malformed OpenSSH output
        mock_private_key = Mock()
        mock_public_key = Mock()
        mock_private_key.public_key.return_value = mock_public_key
        mock_public_key.public_bytes.return_value = b"malformed"
        
        with pytest.raises(ValueError, match="Malformed OpenSSH public key"):
            auth.get_public_key_b64(mock_private_key)

    def test_sign_request_basic(self, temp_key_pair):
        """Test basic request signing functionality."""
        temp_key_path, private_key = temp_key_pair
        
        auth = OllamaAuth(key_path=temp_key_path)
        
        method = "POST"
        path = "/api/chat"
        
        with patch('time.time', return_value=1234567890):
            auth_token, timestamp = auth.sign_request(method, path)
        
        assert timestamp == "1234567890"
        assert isinstance(auth_token, str)
        assert ':' in auth_token  # Should contain public_key:signature format
        
        # Split and verify format
        parts = auth_token.split(':')
        assert len(parts) == 2
        
        public_key_b64, signature_b64 = parts
        
        # Verify public key is valid base64
        try:
            base64.b64decode(public_key_b64)
        except Exception:
            pytest.fail("Public key part is not valid base64")
        
        # Verify signature is valid base64
        try:
            signature_bytes = base64.b64decode(signature_b64)
            assert len(signature_bytes) > 0
        except Exception:
            pytest.fail("Signature part is not valid base64")

    def test_sign_request_with_query_params(self, temp_key_pair):
        """Test request signing with existing query parameters."""
        temp_key_path, private_key = temp_key_pair
        
        auth = OllamaAuth(key_path=temp_key_path)
        
        method = "GET"
        path = "/api/models?format=json"
        
        with patch('time.time', return_value=1234567890):
            auth_token, timestamp = auth.sign_request(method, path)
        
        assert timestamp == "1234567890"
        assert isinstance(auth_token, str)
        
        # The challenge should be "GET,/api/models?format=json&ts=1234567890"
        # We can't easily verify the exact signature, but we can verify format
        parts = auth_token.split(':')
        assert len(parts) == 2

    def test_sign_request_different_methods(self, temp_key_pair):
        """Test request signing with different HTTP methods."""
        temp_key_path, private_key = temp_key_pair
        
        auth = OllamaAuth(key_path=temp_key_path)
        
        methods = ["GET", "POST", "PUT", "DELETE"]
        path = "/api/test"
        
        signatures = {}
        
        for method in methods:
            with patch('time.time', return_value=1234567890):
                auth_token, timestamp = auth.sign_request(method, path)
            signatures[method] = auth_token
        
        # All signatures should be different (different challenges)
        unique_signatures = set(signatures.values())
        assert len(unique_signatures) == len(methods)

    def test_sign_request_different_paths(self, temp_key_pair):
        """Test request signing with different paths."""
        temp_key_path, private_key = temp_key_pair
        
        auth = OllamaAuth(key_path=temp_key_path)
        
        method = "POST"
        paths = ["/api/chat", "/api/generate", "/api/models"]
        
        signatures = {}
        
        for path in paths:
            with patch('time.time', return_value=1234567890):
                auth_token, timestamp = auth.sign_request(method, path)
            signatures[path] = auth_token
        
        # All signatures should be different (different challenges)
        unique_signatures = set(signatures.values())
        assert len(unique_signatures) == len(paths)

    def test_sign_request_file_not_found(self):
        """Test request signing when key file doesn't exist."""
        auth = OllamaAuth(key_path="/nonexistent/path/key")
        
        with pytest.raises(FileNotFoundError, match="Could not find Ollama private key"):
            auth.sign_request("POST", "/api/chat")

    def test_sign_request_invalid_key(self, invalid_key_file):
        """Test request signing with invalid key file."""
        auth = OllamaAuth(key_path=invalid_key_file)
        
        with pytest.raises(ValueError, match="Invalid private key"):
            auth.sign_request("POST", "/api/chat")

    @patch('time.time')
    def test_sign_request_timestamp_generation(self, mock_time, temp_key_pair):
        """Test that timestamps are generated correctly."""
        temp_key_path, private_key = temp_key_pair
        
        auth = OllamaAuth(key_path=temp_key_path)
        
        # Test with different timestamps
        mock_time.return_value = 1000.5
        _, timestamp1 = auth.sign_request("POST", "/api/chat")
        assert timestamp1 == "1000"
        
        mock_time.return_value = 2000.9
        _, timestamp2 = auth.sign_request("POST", "/api/chat")
        assert timestamp2 == "2000"

    def test_signature_verification_challenge_format(self, temp_key_pair):
        """Test that the challenge is formatted correctly for signature verification."""
        temp_key_path, private_key = temp_key_pair
        
        auth = OllamaAuth(key_path=temp_key_path)
        
        method = "POST"
        path = "/api/chat"
        
        with patch('time.time', return_value=1234567890):
            auth_token, timestamp = auth.sign_request(method, path)
        
        # Extract signature and verify it was created with correct challenge
        public_key_b64, signature_b64 = auth_token.split(':')
        signature = base64.b64decode(signature_b64)
        
        # The challenge should be "POST,/api/chat?ts=1234567890"
        expected_challenge = f"{method},{path}?ts={timestamp}"
        
        # Load the private key and get public key for verification
        loaded_private_key = auth.load_private_key()
        public_key = loaded_private_key.public_key()
        
        # Verify the signature
        try:
            public_key.verify(signature, expected_challenge.encode())
            # If no exception is raised, signature is valid
        except Exception:
            pytest.fail("Signature verification failed - challenge format incorrect")

    def test_integration_full_flow(self, temp_key_pair):
        """Test the complete authentication flow integration."""
        temp_key_path, private_key = temp_key_pair
        
        # Test complete flow: init -> sign -> verify
        auth = OllamaAuth(key_path=temp_key_path)
        
        method = "POST"
        path = "/api/chat"
        
        # Sign a request
        auth_token, timestamp = auth.sign_request(method, path)
        
        # Verify the components
        assert isinstance(auth_token, str)
        assert isinstance(timestamp, str)
        assert int(timestamp) > 0  # Should be a valid timestamp
        
        # Verify token format
        parts = auth_token.split(':')
        assert len(parts) == 2
        
        public_key_b64, signature_b64 = parts
        
        # Verify we can decode both parts
        public_key_bytes = base64.b64decode(public_key_b64)
        signature_bytes = base64.b64decode(signature_b64)
        
        assert len(public_key_bytes) > 0
        assert len(signature_bytes) > 0
        
        # Verify the signature is valid for the challenge
        challenge = f"{method},{path}?ts={timestamp}"
        loaded_private_key = auth.load_private_key()
        public_key = loaded_private_key.public_key()
        
        # This should not raise an exception
        public_key.verify(signature_bytes, challenge.encode())
