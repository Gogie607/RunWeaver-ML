from __future__ import annotations

from collections.abc import Mapping

import numpy as np


_PATH_SEPARATOR = "/"


def encode_nested_npz(payload: Mapping[str, object]) -> dict[str, np.ndarray]:
    """
    Convert a nested dict into flat NPZ entries without object arrays.

    This flattening is only an on-disk encoding detail. Calling
    decode_nested_npz() restores the exact nested runtime structure.
    """
    encoded: dict[str, np.ndarray] = {}

    def visit(prefix: str, value: object) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                path = f"{prefix}{_PATH_SEPARATOR}{key}" if prefix else str(key)
                visit(path, child)
            return

        encoded[prefix] = np.asarray(value)

    visit("", payload)
    return encoded



def decode_nested_npz(payload: Mapping[str, object]) -> dict:
    """Restore a canonical nested payload from NPZ path keys."""
    decoded: dict = {}

    for path, value in payload.items():
        parts = str(path).split(_PATH_SEPARATOR)
        node = decoded

        for part in parts[:-1]:
            node = node.setdefault(part, {})

        node[parts[-1]] = np.asarray(value)

    return decoded
