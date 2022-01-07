import json
import os
import time
from pathlib import Path

from util.config import get_project_root


def export(data, name):
    time_str = time.strftime("%Y%m%d-%H%M%S")
    dirpath = os.path.join(get_project_root(), 'backup', 'deleted_buckets')
    Path(dirpath).mkdir(parents=True, exist_ok=True)
    filepath = os.path.join(dirpath, time_str + '-' + name + '.json')
    with open(filepath, 'a+') as file:
        json.dump(data, file)
