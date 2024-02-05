import json


class RequestError(Exception):
    """
    Common class for request errors.
    """

    def __init__(self, error: str):
        super().__init__(error)
        self.error = error
        'Reason for the error.'


class ResponseError(Exception):
    """
    Common class for response errors.
    """

    def __init__(self, error: str, status_code: int = -1):
        try:
            # try to parse content as JSON and extract 'error'
            # fallback to raw content if JSON parsing fails
            error = json.loads(error).get('error', error)
        except json.JSONDecodeError:
            ...

        super().__init__(error)
        self.error = error
        'Reason for the error.'

        self.status_code = status_code
        'HTTP status code of the response.'
