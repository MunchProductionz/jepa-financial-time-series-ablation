"""Small dotted-path import helpers used by builders."""

from __future__ import annotations

import importlib
from typing import Any


def locate(path: str) -> Any:
    """Resolve ``package.module:object`` or ``package.module.object``."""

    if ":" in path:
        module_name, object_name = path.split(":", 1)
    else:
        module_name, object_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    target: Any = module
    for part in object_name.split("."):
        target = getattr(target, part)
    return target


def instantiate(path: str, *args: Any, **kwargs: Any) -> Any:
    return locate(path)(*args, **kwargs)

