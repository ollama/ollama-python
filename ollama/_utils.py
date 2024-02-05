import binascii
import io
import urllib.parse
from base64 import b64encode, b64decode
from os import PathLike
from pathlib import Path
from typing import Optional, Union, Any

from ollama._errors import RequestError


def _encode_image(image) -> str:
    """
    >>> _encode_image(b'ollama')
    'b2xsYW1h'
    >>> _encode_image(io.BytesIO(b'ollama'))
    'b2xsYW1h'
    >>> _encode_image('LICENSE')
    'TUlUIExpY2Vuc2UKCkNvcHlyaWdodCAoYykgT2xsYW1hCgpQZXJtaXNzaW9uIGlzIGhlcmVieSBncmFudGVkLCBmcmVlIG9mIGNoYXJnZSwgdG8gYW55IHBlcnNvbiBvYnRhaW5pbmcgYSBjb3B5Cm9mIHRoaXMgc29mdHdhcmUgYW5kIGFzc29jaWF0ZWQgZG9jdW1lbnRhdGlvbiBmaWxlcyAodGhlICJTb2Z0d2FyZSIpLCB0byBkZWFsCmluIHRoZSBTb2Z0d2FyZSB3aXRob3V0IHJlc3RyaWN0aW9uLCBpbmNsdWRpbmcgd2l0aG91dCBsaW1pdGF0aW9uIHRoZSByaWdodHMKdG8gdXNlLCBjb3B5LCBtb2RpZnksIG1lcmdlLCBwdWJsaXNoLCBkaXN0cmlidXRlLCBzdWJsaWNlbnNlLCBhbmQvb3Igc2VsbApjb3BpZXMgb2YgdGhlIFNvZnR3YXJlLCBhbmQgdG8gcGVybWl0IHBlcnNvbnMgdG8gd2hvbSB0aGUgU29mdHdhcmUgaXMKZnVybmlzaGVkIHRvIGRvIHNvLCBzdWJqZWN0IHRvIHRoZSBmb2xsb3dpbmcgY29uZGl0aW9uczoKClRoZSBhYm92ZSBjb3B5cmlnaHQgbm90aWNlIGFuZCB0aGlzIHBlcm1pc3Npb24gbm90aWNlIHNoYWxsIGJlIGluY2x1ZGVkIGluIGFsbApjb3BpZXMgb3Igc3Vic3RhbnRpYWwgcG9ydGlvbnMgb2YgdGhlIFNvZnR3YXJlLgoKVEhFIFNPRlRXQVJFIElTIFBST1ZJREVEICJBUyBJUyIsIFdJVEhPVVQgV0FSUkFOVFkgT0YgQU5ZIEtJTkQsIEVYUFJFU1MgT1IKSU1QTElFRCwgSU5DTFVESU5HIEJVVCBOT1QgTElNSVRFRCBUTyBUSEUgV0FSUkFOVElFUyBPRiBNRVJDSEFOVEFCSUxJVFksCkZJVE5FU1MgRk9SIEEgUEFSVElDVUxBUiBQVVJQT1NFIEFORCBOT05JTkZSSU5HRU1FTlQuIElOIE5PIEVWRU5UIFNIQUxMIFRIRQpBVVRIT1JTIE9SIENPUFlSSUdIVCBIT0xERVJTIEJFIExJQUJMRSBGT1IgQU5ZIENMQUlNLCBEQU1BR0VTIE9SIE9USEVSCkxJQUJJTElUWSwgV0hFVEhFUiBJTiBBTiBBQ1RJT04gT0YgQ09OVFJBQ1QsIFRPUlQgT1IgT1RIRVJXSVNFLCBBUklTSU5HIEZST00sCk9VVCBPRiBPUiBJTiBDT05ORUNUSU9OIFdJVEggVEhFIFNPRlRXQVJFIE9SIFRIRSBVU0UgT1IgT1RIRVIgREVBTElOR1MgSU4gVEhFClNPRlRXQVJFLgo='
    >>> _encode_image(Path('LICENSE'))
    'TUlUIExpY2Vuc2UKCkNvcHlyaWdodCAoYykgT2xsYW1hCgpQZXJtaXNzaW9uIGlzIGhlcmVieSBncmFudGVkLCBmcmVlIG9mIGNoYXJnZSwgdG8gYW55IHBlcnNvbiBvYnRhaW5pbmcgYSBjb3B5Cm9mIHRoaXMgc29mdHdhcmUgYW5kIGFzc29jaWF0ZWQgZG9jdW1lbnRhdGlvbiBmaWxlcyAodGhlICJTb2Z0d2FyZSIpLCB0byBkZWFsCmluIHRoZSBTb2Z0d2FyZSB3aXRob3V0IHJlc3RyaWN0aW9uLCBpbmNsdWRpbmcgd2l0aG91dCBsaW1pdGF0aW9uIHRoZSByaWdodHMKdG8gdXNlLCBjb3B5LCBtb2RpZnksIG1lcmdlLCBwdWJsaXNoLCBkaXN0cmlidXRlLCBzdWJsaWNlbnNlLCBhbmQvb3Igc2VsbApjb3BpZXMgb2YgdGhlIFNvZnR3YXJlLCBhbmQgdG8gcGVybWl0IHBlcnNvbnMgdG8gd2hvbSB0aGUgU29mdHdhcmUgaXMKZnVybmlzaGVkIHRvIGRvIHNvLCBzdWJqZWN0IHRvIHRoZSBmb2xsb3dpbmcgY29uZGl0aW9uczoKClRoZSBhYm92ZSBjb3B5cmlnaHQgbm90aWNlIGFuZCB0aGlzIHBlcm1pc3Npb24gbm90aWNlIHNoYWxsIGJlIGluY2x1ZGVkIGluIGFsbApjb3BpZXMgb3Igc3Vic3RhbnRpYWwgcG9ydGlvbnMgb2YgdGhlIFNvZnR3YXJlLgoKVEhFIFNPRlRXQVJFIElTIFBST1ZJREVEICJBUyBJUyIsIFdJVEhPVVQgV0FSUkFOVFkgT0YgQU5ZIEtJTkQsIEVYUFJFU1MgT1IKSU1QTElFRCwgSU5DTFVESU5HIEJVVCBOT1QgTElNSVRFRCBUTyBUSEUgV0FSUkFOVElFUyBPRiBNRVJDSEFOVEFCSUxJVFksCkZJVE5FU1MgRk9SIEEgUEFSVElDVUxBUiBQVVJQT1NFIEFORCBOT05JTkZSSU5HRU1FTlQuIElOIE5PIEVWRU5UIFNIQUxMIFRIRQpBVVRIT1JTIE9SIENPUFlSSUdIVCBIT0xERVJTIEJFIExJQUJMRSBGT1IgQU5ZIENMQUlNLCBEQU1BR0VTIE9SIE9USEVSCkxJQUJJTElUWSwgV0hFVEhFUiBJTiBBTiBBQ1RJT04gT0YgQ09OVFJBQ1QsIFRPUlQgT1IgT1RIRVJXSVNFLCBBUklTSU5HIEZST00sCk9VVCBPRiBPUiBJTiBDT05ORUNUSU9OIFdJVEggVEhFIFNPRlRXQVJFIE9SIFRIRSBVU0UgT1IgT1RIRVIgREVBTElOR1MgSU4gVEhFClNPRlRXQVJFLgo='
    >>> _encode_image('YWJj')
    'YWJj'
    >>> _encode_image(b'YWJj')
    'YWJj'
    """

    if p := _as_path(image):
        return b64encode(p.read_bytes()).decode('utf-8')

    try:
        b64decode(image, validate=True)
        return image if isinstance(image, str) else image.decode('utf-8')
    except (binascii.Error, TypeError):
        ...

    if b := _as_bytesio(image):
        return b64encode(b.read()).decode('utf-8')

    raise RequestError('image must be bytes, path-like object, or file-like object')


def _as_path(s: Optional[Union[str, PathLike]]) -> Union[Path, None]:
    if isinstance(s, str) or isinstance(s, Path):
        try:
            if (p := Path(s)).exists():
                return p
        except Exception:
            ...
    return None


def _as_bytesio(s: Any) -> Union[io.BytesIO, None]:
    if isinstance(s, io.BytesIO):
        return s
    elif isinstance(s, bytes):
        return io.BytesIO(s)
    return None
def _parse_host(host: Optional[str]) -> str:
    """
    >>> _parse_host(None)
    'http://127.0.0.1:11434'
    >>> _parse_host('')
    'http://127.0.0.1:11434'
    >>> _parse_host('1.2.3.4')
    'http://1.2.3.4:11434'
    >>> _parse_host(':56789')
    'http://127.0.0.1:56789'
    >>> _parse_host('1.2.3.4:56789')
    'http://1.2.3.4:56789'
    >>> _parse_host('http://1.2.3.4')
    'http://1.2.3.4:80'
    >>> _parse_host('https://1.2.3.4')
    'https://1.2.3.4:443'
    >>> _parse_host('https://1.2.3.4:56789')
    'https://1.2.3.4:56789'
    >>> _parse_host('example.com')
    'http://example.com:11434'
    >>> _parse_host('example.com:56789')
    'http://example.com:56789'
    >>> _parse_host('http://example.com')
    'http://example.com:80'
    >>> _parse_host('https://example.com')
    'https://example.com:443'
    >>> _parse_host('https://example.com:56789')
    'https://example.com:56789'
    >>> _parse_host('example.com/')
    'http://example.com:11434'
    >>> _parse_host('example.com:56789/')
    'http://example.com:56789'
    """

    host, port = host or '', 11434
    scheme, _, hostport = host.partition('://')
    if not hostport:
        scheme, hostport = 'http', host
    elif scheme == 'http':
        port = 80
    elif scheme == 'https':
        port = 443

    split = urllib.parse.urlsplit('://'.join([scheme, hostport]))
    host = split.hostname or '127.0.0.1'
    port = split.port or port

    return f'{scheme}://{host}:{port}'
