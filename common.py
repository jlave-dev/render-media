"""Small, dependency-free archive primitives shared by the CLI and recorder."""
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.writing-')
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def slug(value):
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,79}', value):
        raise ValueError('Use 1-80 lowercase letters, digits, underscores or hyphens')
    return value


def token(value):
    return value if re.fullmatch(r'[A-Za-z0-9_-]{1,80}', value) else hashlib.sha256(value.encode()).hexdigest()[:16]


def extension(path):
    suffix = Path(path).suffix.lower()
    return suffix if re.fullmatch(r'\.[a-z0-9]{1,16}', suffix) else '.bin'


def contained(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Path escapes its root: ' + str(name))
    return path


def graph_value(graph, binding, value):
    node, field = binding.split('.', 1)
    inputs = graph[node]['inputs']
    if field not in inputs:
        raise ValueError('No graph input: ' + binding)
    inputs[field] = value


def leaves(value, prefix=''):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from leaves(child, prefix + '.' + key if prefix else key)
    elif isinstance(value, str):
        yield prefix, value
