"""Независимые от реализации проверки инвариантов разбиения."""

from __future__ import annotations

import numpy as np
import pandas as pd


def keys_of(series: pd.Series) -> np.ndarray:
    return series.array.view("i8")


def reference_bounds(sorted_keys: np.ndarray, chunk_size: int) -> list[tuple[int, int]]:
    """Наивный эталон: копим целые группы одинаковых дат, пока не наберём chunk_size."""
    _, counts = np.unique(sorted_keys, return_counts=True)
    bounds, start, size = [], 0, 0
    for count in counts:
        size += int(count)
        if size >= chunk_size:
            bounds.append((start, start + size))
            start, size = start + size, 0
    if size:
        bounds.append((start, start + size))
    return bounds


def assert_valid_bounds(sorted_keys: np.ndarray, bounds: list[tuple[int, int]], chunk_size: int) -> None:
    """Векторная проверка границ чанков над отсортированными ключами — годится для 1M+ строк."""
    n = len(sorted_keys)
    if n == 0:
        assert bounds == []
        return
    starts = np.fromiter((b[0] for b in bounds), dtype=np.int64, count=len(bounds))
    stops = np.fromiter((b[1] for b in bounds), dtype=np.int64, count=len(bounds))
    sizes = stops - starts

    # Чанки покрывают все строки ровно один раз, без дыр и наложений.
    assert starts[0] == 0 and stops[-1] == n
    assert np.array_equal(starts[1:], stops[:-1])
    assert (sizes > 0).all()
    # Все чанки, кроме хвоста, не меньше желаемого размера.
    assert (sizes[:-1] >= chunk_size).all()
    # Граница никогда не рвёт группу одинаковых дат.
    assert (sorted_keys[stops[:-1] - 1] < sorted_keys[stops[:-1]]).all()
    # Минимальность: без последней группы чанк был бы меньше chunk_size.
    last_group_starts = np.searchsorted(sorted_keys, sorted_keys[stops - 1], side="left")
    assert (last_group_starts - starts < chunk_size).all()


def assert_valid_chunks(
    df: pd.DataFrame, chunks: list[pd.DataFrame], column: str, chunk_size: int
) -> None:
    """Проверяет готовые чанки-фреймы на все требования задачи."""
    if len(df) == 0:
        assert chunks == []
        return

    sizes = [len(chunk) for chunk in chunks]
    assert sum(sizes) == len(df)
    assert all(size >= chunk_size for size in sizes[:-1])

    chunk_keys = [keys_of(chunk[column]) for chunk in chunks]
    for keys in chunk_keys:
        assert len(keys) > 0
        assert (keys[1:] >= keys[:-1]).all(), "внутри чанка даты идут по возрастанию"
    for prev, nxt in zip(chunk_keys, chunk_keys[1:]):
        assert prev[-1] < nxt[0], "даты соседних чанков не пересекаются"
    for keys in chunk_keys:
        last_group_size = len(keys) - np.searchsorted(keys, keys[-1], side="left")
        assert len(keys) - last_group_size < chunk_size, "чанк не берёт лишнюю группу"

    # Каждая строка исходного фрейма встречается ровно в одном чанке и не искажена
    # (индекс df должен быть уникальным).
    combined = pd.concat(chunks).sort_index(kind="stable")
    pd.testing.assert_frame_equal(combined, df.sort_index(kind="stable"))
