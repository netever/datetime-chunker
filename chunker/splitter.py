"""Разбиение DataFrame на чанки по datetime-колонке.

Бизнес-правило: все строки с одинаковым datetime попадают в один чанк,
а диапазоны дат разных чанков не пересекаются.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Iterator

import numpy as np
import pandas as pd

__all__ = ["chunk_bounds", "split_by_datetime", "split_stream"]

# Проверка сортировки идёт блоками: временная bool-маска не больше 64 KiB.
_SORT_CHECK_BLOCK = 1 << 16


def split_by_datetime(
    df: pd.DataFrame,
    column: Hashable,
    chunk_size: int,
    *,
    assume_sorted: bool = False,
) -> Iterator[pd.DataFrame]:
    """Лениво разбивает ``df`` на чанки по колонке ``column``.

    Гарантии:
      * все повторения одного datetime лежат в одном чанке;
      * даты разных чанков не пересекаются, чанки идут по возрастанию дат;
      * каждый чанк, кроме, возможно, последнего (хвоста), содержит
        ``>= chunk_size`` строк и при этом минимален: в него не берётся
        лишняя группа одинаковых дат.

    Если колонка уже отсортирована по возрастанию, чанки — это срезы
    ``df.iloc[start:stop]``, т.е. view без копирования данных: дополнительная
    память O(1). Сортированность проверяется за O(n); ``assume_sorted=True``
    пропускает проверку (на несортированных данных результат будет неверным).

    Иначе строится стабильный argsort (O(n log n), 8 байт на строку), и чанки
    собираются через ``df.take`` по требованию — в памяти одновременно
    существует только копия текущего чанка.

    NaT считается значением меньше любой даты: все NaT образуют одну группу
    в самом первом чанке.
    """
    _validate_chunk_size(chunk_size)
    keys = _int64_keys(df[column])
    if assume_sorted or _is_sorted(keys):
        return (df.iloc[start:stop] for start, stop in _iter_bounds(keys, chunk_size))
    return _iter_unsorted(df, keys, chunk_size)


def split_stream(
    batches: Iterable[pd.DataFrame],
    column: Hashable,
    chunk_size: int,
) -> Iterator[pd.DataFrame]:
    """Потоковый вариант ``split_by_datetime`` для данных, которые не помещаются в память.

    ``batches`` — любой источник DataFrame-пачек: ``pd.read_csv(..., chunksize=N)``,
    parquet по row group'ам, курсор БД с ``ORDER BY``. Пачки должны быть отсортированы
    по ``column`` и идти друг за другом по времени; группа одинаковых дат может
    разрываться между пачками — она склеивается. Результат совпадает
    с ``split_by_datetime`` на склеенном фрейме.

    Память: O(batch_size + chunk_size) независимо от общего числа строк — в памяти
    текущая пачка и строки незавершённого чанка. Чанк внутри одной пачки — view,
    чанк на стыке пачек собирается ``pd.concat`` (копия размером с чанк).

    Время: O(n) на проверку порядка (потоку нельзя доверять на слово)
    плюс один бинарный поиск на чанк. Неотсортированный поток — ``ValueError``.
    """
    _validate_chunk_size(chunk_size)
    return _iter_stream(batches, column, chunk_size)


def _iter_stream(batches: Iterable[pd.DataFrame], column: Hashable, chunk_size: int) -> Iterator[pd.DataFrame]:
    pending: list[pd.DataFrame] = []  # строки незавершённого чанка из прошлых пачек
    pending_len = 0
    last_key = None  # последняя дата потока; ею заканчивается pending
    for batch in batches:
        keys = _int64_keys(batch[column])
        n = len(keys)
        if n == 0:
            continue
        if not _is_sorted(keys) or (last_key is not None and keys[0] < last_key):
            raise ValueError(f"Batches must be sorted by {column!r} ascending, within and across batches")

        offset = 0
        while offset < n:
            need = chunk_size - pending_len
            if need > 0:
                pivot = offset + need - 1
                if pivot >= n:
                    break  # строк не хватает — ждём следующую пачку
                value = keys[pivot]
            else:
                # pending уже набрал chunk_size, но его последняя группа могла продолжиться здесь.
                pivot, value = offset, last_key
            stop = pivot + int(np.searchsorted(keys[pivot:], value, side="right"))
            if stop == n:
                break  # группа доходит до конца пачки и может продолжиться в следующей
            if stop > offset:
                pending.append(batch.iloc[offset:stop])
            yield _take_all(pending)
            pending_len = 0
            offset = stop

        if offset < n:
            pending.append(batch.iloc[offset:])
            pending_len += n - offset
        last_key = keys[-1]

    if pending:
        yield _take_all(pending)


def _take_all(pieces: list[pd.DataFrame]) -> pd.DataFrame:
    """Склеивает куски и очищает список: пока чанк обрабатывают, генератор не держит их копию."""
    chunk = pieces[0] if len(pieces) == 1 else pd.concat(pieces)
    pieces.clear()
    return chunk


def chunk_bounds(keys: np.ndarray, chunk_size: int) -> Iterator[tuple[int, int]]:
    """Лениво отдаёт позиции ``(start, stop)`` чанков для отсортированного ``keys``.

    На каждый чанк — один бинарный поиск: O(k·log n) для k чанков,
    дополнительная память O(1).
    """
    _validate_chunk_size(chunk_size)
    return _iter_bounds(keys, chunk_size)


def _iter_bounds(keys: np.ndarray, chunk_size: int) -> Iterator[tuple[int, int]]:
    n = len(keys)
    start = 0
    while start < n:
        # Чтобы набрать chunk_size строк, чанк обязан включить позицию pivot...
        pivot = start + chunk_size - 1
        if pivot >= n - 1:
            yield start, n
            return
        # ...и дотянуться до конца группы значения keys[pivot].
        stop = pivot + int(np.searchsorted(keys[pivot:], keys[pivot], side="right"))
        yield start, stop
        start = stop


def _iter_unsorted(df: pd.DataFrame, keys: np.ndarray, chunk_size: int) -> Iterator[pd.DataFrame]:
    order = np.argsort(keys, kind="stable")  # stable: внутри группы сохраняется исходный порядок
    sorted_keys = keys[order]
    stops = np.fromiter((stop for _, stop in _iter_bounds(sorted_keys, chunk_size)), dtype=np.intp)
    del sorted_keys  # пока отдаются чанки, нужен только order
    start = 0
    for stop in stops:
        yield df.take(order[start:stop])
        start = stop


def _int64_keys(series: pd.Series) -> np.ndarray:
    """int64-view колонки без копирования (naive/tz-aware, любая единица времени)."""
    if not pd.api.types.is_datetime64_any_dtype(series.dtype):
        raise TypeError(
            f"Column {series.name!r} must have datetime64 dtype, got {series.dtype}; "
            "convert it with pd.to_datetime first"
        )
    return series.array.view("i8")


def _is_sorted(keys: np.ndarray) -> bool:
    """O(n), память ограничена одним блоком, выход на первом нарушении порядка."""
    for start in range(0, len(keys) - 1, _SORT_CHECK_BLOCK):
        block = keys[start : start + _SORT_CHECK_BLOCK + 1]  # +1: стык с соседним блоком
        if (block[1:] < block[:-1]).any():
            return False
    return True


def _validate_chunk_size(chunk_size: int) -> None:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, (int, np.integer)):
        raise TypeError(f"chunk_size must be an integer, got {type(chunk_size).__name__}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
