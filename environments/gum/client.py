# gum http client interface

import re
import requests
import subprocess
from enum import Enum

HOST = 'localhost'
PORT = 3000

REGEX_ERROR = r'^(ERR_\w+): (.*)$'

class ErrorType(Enum):
    UNKNOWN = 'UNKNOWN'
    NOCODE = 'NOCODE'
    NORETURN = 'NORETURN'
    NOELEMENT = 'NOELEMENT'
    PARSE = 'PARSE'
    GENERATE = 'GENERATE'
    RENDER = 'RENDER'

class GumError(Exception):
    def __init__(self, error_type, error_message):
        self.error_type = error_type
        self.error_message = error_message
        super().__init__(f'{error_type.name}: {error_message}')

def parse_error(text):
    # check for error message
    regex = REGEX_ERROR if type(text) is str else REGEX_ERROR.encode()
    match = re.match(regex, text)
    if not match:
        return text

    # get error type and message
    error_type = match.group(1)
    error_message = match.group(2)

    # return error type
    if error_type == 'ERR_NOCODE':
        return GumError(ErrorType.NOCODE, error_message)
    elif error_type == 'ERR_NORETURN':
        return GumError(ErrorType.NORETURN, error_message)
    elif error_type == 'ERR_NOELEMENT':
        return GumError(ErrorType.NOELEMENT, error_message)
    elif error_type == 'ERR_PARSE':
        return GumError(ErrorType.PARSE, error_message)
    elif error_type == 'ERR_GENERATE':
        return GumError(ErrorType.GENERATE, error_message)
    elif error_type == 'ERR_RENDER':
        return GumError(ErrorType.RENDER, error_message)
    elif error_type == 'ERR_UNKNOWN':
        return GumError(ErrorType.UNKNOWN, error_message)
    else:
        return GumError(ErrorType.UNKNOWN, error_message)

class GumClient:
    def __init__(self):
        args = [ 'node', 'server.ts', '--host', HOST, '--port', str(PORT) ]
        self.server = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def __del__(self):
        self.server.terminate()

    def evaluate(self, code, size=1024):
        url = f'http://{HOST}:{PORT}/eval?size={size}'
        response = requests.post(url, data=code, headers={'Content-Type': 'text/plain'})
        return parse_error(response.text)

    def render(self, code, size=1024):
        url = f'http://{HOST}:{PORT}/render?size={size}'
        response = requests.post(url, data=code, headers={'Content-Type': 'text/plain'})
        return parse_error(response.content)
