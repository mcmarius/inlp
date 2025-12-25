import os
from pathlib import Path


def path_resolver(path, external=False):
    if 'notebooks' in os.path.abspath(os.path.curdir):
        if external:
            return ".." / Path(path)
        Path.mkdir(Path(path).parent, parents=True, exist_ok=True)
        return Path(path)

    if external:
        return Path(path)

    notebooks_path = Path('notebooks') / path
    Path.mkdir(Path(notebooks_path).parent, parents=True, exist_ok=True)
    return notebooks_path
