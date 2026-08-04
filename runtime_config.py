import os
import re


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
INSTANCE_NAME = os.getenv('STT_INSTANCE', '').strip()
SHARED_DATA_ROOT = os.path.join(PROJECT_DIR, 'data')

if INSTANCE_NAME and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,31}', INSTANCE_NAME):
    raise RuntimeError('STT_INSTANCE must use 1-32 letters, numbers, underscores, or hyphens.')

DATA_ROOT = (
    os.path.join(SHARED_DATA_ROOT, 'instances', INSTANCE_NAME)
    if INSTANCE_NAME
    else SHARED_DATA_ROOT
)


def server_port():
    try:
        port = int(os.getenv('STT_PORT', '5000'))
    except ValueError as error:
        raise RuntimeError('STT_PORT must be an integer.') from error
    if not 1 <= port <= 65535:
        raise RuntimeError('STT_PORT must be between 1 and 65535.')
    return port
