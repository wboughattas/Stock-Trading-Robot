import os
from pathlib import Path
from dotenv import load_dotenv


def get_project_root() -> Path:
    return Path(__file__).parent.parent  # Project root


def get_env():
    dotenv_path = os.path.join(get_project_root(), 'venv', '.env')
    load_dotenv(dotenv_path)

    # to connect to Finnhub websocket api
    _socket = os.getenv('socket')

    # to connect to an already-set-up local SQL server
    _token = os.getenv('token')
    _org = os.getenv('org')

    return _socket, _token, _org
