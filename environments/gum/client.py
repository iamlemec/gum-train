# gum http client interface

import re
import gum
from enum import Enum

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

def renderGum(code, size=512):
    # run the render
    text = gum.render(code, pixels=size)

    # check for error message
    regex = REGEX_ERROR if type(text) is str else REGEX_ERROR.encode()
    match = re.match(regex, text)
    if not match:
        return text

    # get error type and message
    error_type = match.group(1)
    error_message = match.group(2)

    # raise error
    if error_type == 'ERR_NOCODE':
        raise GumError(ErrorType.NOCODE, error_message)
    elif error_type == 'ERR_NORETURN':
        raise GumError(ErrorType.NORETURN, error_message)
    elif error_type == 'ERR_NOELEMENT':
        raise GumError(ErrorType.NOELEMENT, error_message)
    elif error_type == 'ERR_PARSE':
        raise GumError(ErrorType.PARSE, error_message)
    elif error_type == 'ERR_GENERATE':
        raise GumError(ErrorType.GENERATE, error_message)
    elif error_type == 'ERR_RENDER':
        raise GumError(ErrorType.RENDER, error_message)
    elif error_type == 'ERR_UNKNOWN':
        raise GumError(ErrorType.UNKNOWN, error_message)
    else:
        raise GumError(ErrorType.UNKNOWN, error_message)
