"""Генерация тестовых данных: datetime-колонка с повторениями."""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["generate_frame"]


def generate_frame(
    n_rows: int = 1_000_000,
    *,
    max_repeats: int = 5,
    start: str = "2023-01-01",
    step: str = "1s",
    shuffle: bool = False,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Фрейм из ``n_rows`` строк, каждая дата повторяется от 1 до ``max_repeats`` раз.

    Колонки:
      * ``dt`` — datetime64[ns], по возрастанию (если не ``shuffle``);
      * ``id`` — int64, номер строки в отсортированном порядке: по нему удобно
        проверять, что ни одна строка не потерялась и не задублировалась;
      * ``value`` — float32, полезная нагрузка.
    """
    if n_rows < 0:
        raise ValueError(f"n_rows must be >= 0, got {n_rows}")
    if max_repeats < 1:
        raise ValueError(f"max_repeats must be >= 1, got {max_repeats}")

    rng = np.random.default_rng(seed)
    mean_repeats = (1 + max_repeats) / 2
    counts = rng.integers(1, max_repeats + 1, size=int(n_rows / mean_repeats * 1.05) + 16)
    while counts.sum() < n_rows:  # на практике не срабатывает: запас 5% выше
        counts = np.concatenate([counts, rng.integers(1, max_repeats + 1, size=len(counts))])
    # Обрезаем повторы ровно до n_rows, чтобы np.repeat не выделял лишнего.
    counts = counts[: np.searchsorted(np.cumsum(counts), n_rows) + 1]
    counts[-1] -= counts.sum() - n_rows

    # Арифметика in-place, без копий: пик памяти близок к размеру итогового фрейма.
    dt = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    del counts
    dt *= pd.Timedelta(step).value
    dt += pd.Timestamp(start).value
    dt = dt.view("datetime64[ns]")
    ids = np.arange(n_rows, dtype=np.int64)
    value = rng.random(n_rows, dtype=np.float32)

    if shuffle:
        perm = rng.permutation(n_rows)
        # По одной колонке: в памяти не бывает двух полных копий фрейма сразу.
        dt = dt[perm]
        ids = ids[perm]
        value = value[perm]
        del perm

    return pd.DataFrame({"dt": dt, "id": ids, "value": value}, copy=False)
