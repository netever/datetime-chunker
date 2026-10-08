"""Генерация тестовых данных: datetime-колонка с повторениями."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

__all__ = ["generate_batches", "generate_frame"]


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
    _validate(n_rows, max_repeats)
    rng = np.random.default_rng(seed)
    counts, _ = _take_rows(_draw_counts(rng, n_rows, max_repeats), n_rows)
    dt = _timestamps(0, counts, start, step)
    del counts
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


def generate_batches(
    n_rows: int = 1_000_000,
    batch_size: int = 100_000,
    *,
    max_repeats: int = 5,
    start: str = "2023-01-01",
    step: str = "1s",
    seed: int | None = 42,
) -> Iterator[pd.DataFrame]:
    """Отсортированные данные того же вида, что ``generate_frame``, но лениво, пачками.

    В памяти только текущая пачка, поэтому можно сгенерировать хоть миллиард строк.
    Группы одинаковых дат не выравниваются по пачкам и попадают на стыки.
    Индекс и ``id`` сквозные: склеенные пачки дают фрейм с ``RangeIndex(n_rows)``.
    """
    _validate(n_rows, max_repeats)
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    return _iter_batches(n_rows, batch_size, max_repeats, start, step, seed)


def _iter_batches(
    n_rows: int, batch_size: int, max_repeats: int, start: str, step: str, seed: int | None
) -> Iterator[pd.DataFrame]:
    rng = np.random.default_rng(seed)
    group, carry = 0, 0  # первая группа пачки и сколько её строк перешло из прошлой пачки
    for offset in range(0, n_rows, batch_size):
        size = min(batch_size, n_rows - offset)
        counts = _draw_counts(rng, max(size - carry, 0), max_repeats)
        if carry:
            counts = np.concatenate(([carry], counts))
        counts, carry = _take_rows(counts, size)
        dt = _timestamps(group, counts, start, step)
        group += len(counts) - (1 if carry else 0)
        yield pd.DataFrame(
            {
                "dt": dt,
                "id": np.arange(offset, offset + size, dtype=np.int64),
                "value": rng.random(size, dtype=np.float32),
            },
            index=pd.RangeIndex(offset, offset + size),
            copy=False,
        )


def _validate(n_rows: int, max_repeats: int) -> None:
    if n_rows < 0:
        raise ValueError(f"n_rows must be >= 0, got {n_rows}")
    if max_repeats < 1:
        raise ValueError(f"max_repeats must be >= 1, got {max_repeats}")


def _draw_counts(rng: np.random.Generator, n_rows: int, max_repeats: int) -> np.ndarray:
    """Случайные числа повторов дат (1..max_repeats) с суммой не меньше ``n_rows``."""
    mean_repeats = (1 + max_repeats) / 2
    counts = rng.integers(1, max_repeats + 1, size=int(n_rows / mean_repeats * 1.05) + 16)
    while counts.sum() < n_rows:  # на практике не срабатывает: запас 5% выше
        counts = np.concatenate([counts, rng.integers(1, max_repeats + 1, size=len(counts))])
    return counts


def _take_rows(counts: np.ndarray, n_rows: int) -> tuple[np.ndarray, int]:
    """Обрезает повторы ровно до ``n_rows`` строк; возвращает их и остаток последней группы."""
    counts = counts[: np.searchsorted(np.cumsum(counts), n_rows) + 1]
    leftover = int(counts.sum()) - n_rows
    counts[-1] -= leftover
    return counts, leftover


def _timestamps(first_group: int, counts: np.ndarray, start: str, step: str) -> np.ndarray:
    """Даты групп ``first_group, first_group + 1, ...``, каждая повторена ``counts[i]`` раз."""
    # Арифметика in-place, без копий: пик памяти близок к размеру результата.
    dt = np.repeat(np.arange(first_group, first_group + len(counts), dtype=np.int64), counts)
    dt *= pd.Timedelta(step).value
    dt += pd.Timestamp(start).value
    return dt.view("datetime64[ns]")
