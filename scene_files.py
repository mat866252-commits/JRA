"""Utilidades compartidas para archivos nombrados como ``escena_0001.ext``."""

from __future__ import annotations

import re
from pathlib import Path

SCENE_NUMBER = re.compile(r"(?:^|[_\s]?)0*(\d+)$")


def scene_number(path: Path) -> int | None:
    match = SCENE_NUMBER.search(path.stem)
    return int(match.group(1)) if match else None


def indexed_files(folder: str | Path, extensions: set[str], kind: str) -> dict[int, Path]:
    """Devuelve archivos indexados y rechaza nombres ambiguos o duplicados."""
    root = Path(folder)
    if not root.is_dir():
        raise ValueError(f"No existe la carpeta de {kind}: {root}")
    files: dict[int, Path] = {}
    unnamed: list[str] = []
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        number = scene_number(path)
        if number is None:
            unnamed.append(path.name)
        elif number in files:
            raise ValueError(f"Escena {number:04d} repetida en {kind}: {files[number].name} y {path.name}")
        else:
            files[number] = path
    if unnamed:
        raise ValueError(f"Archivos de {kind} sin número de escena: {', '.join(unnamed[:5])}")
    if not files:
        raise ValueError(f"No hay archivos válidos en {root}")
    return files


def numbered(folder: str | Path, extensions: set[str], kind: str) -> set[int]:
    return set(indexed_files(folder, extensions, kind))


def alignment_errors(label: str, expected: set[int], actual: set[int]) -> list[str]:
    missing, extra = sorted(expected - actual), sorted(actual - expected)
    messages = []
    if missing:
        messages.append(f"{label} faltantes: " + ", ".join(f"{number:04d}" for number in missing))
    if extra:
        messages.append(f"{label} sobrantes: " + ", ".join(f"{number:04d}" for number in extra))
    return messages
