import os
from pathlib import Path
from dotenv import load_dotenv
import conf.livedata as livedata  # do not delete
import conf.influxdb as influxdb  # do not delete


def get_project_root() -> Path:
    return Path(__file__).parent.parent  # Project root


def get_env():
    dotenv_path = os.path.join(get_project_root(), 'venv', '.env')
    load_dotenv(dotenv_path)
    pass


def get_conf(*modules):
    book = {}
    for module in modules:
        module = globals().get(module, None)
        if module:
            book.update({key: value for key, value in module.__dict__.items() if
                         not (key.startswith('__') or key.startswith('_'))})
    return book
