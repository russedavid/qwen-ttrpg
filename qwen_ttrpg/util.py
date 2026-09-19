import hashlib
import json
from datetime import datetime, timezone


def packed(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def file_digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()
