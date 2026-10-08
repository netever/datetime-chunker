"""Проверки на сгенерированных данных от 1 000 000 строк."""

import tracemalloc

import numpy as np
import pytest

from chunker import chunk_bounds, generate_frame, split_by_datetime
from tests.conftest import LARGE_ROWS
from tests.helpers import assert_valid_bounds, assert_valid_chunks, keys_of, reference_bounds


def peak_memory(func) -> int:
    """Пиковый объём памяти, выделенной во время func (numpy-буферы tracemalloc видит)."""
    tracemalloc.start()
    try:
        func()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def consume(iterator) -> None:
    for _ in iterator:
        pass


def test_generated_data_size(large_sorted_df, large_shuffled_df):
    assert len(large_sorted_df) == len(large_shuffled_df) == LARGE_ROWS
    assert large_sorted_df["dt"].duplicated().sum() > LARGE_ROWS // 2, "повторов дат много"


@pytest.mark.parametrize("chunk_size", [1, 2, 7, 1_000, 65_537, 999_999, LARGE_ROWS, 2 * LARGE_ROWS])
def test_bounds_match_reference(large_sorted_df, chunk_size):
    keys = keys_of(large_sorted_df["dt"])
    bounds = list(chunk_bounds(keys, chunk_size))
    assert_valid_bounds(keys, bounds, chunk_size)
    assert bounds == reference_bounds(keys, chunk_size)


@pytest.mark.parametrize("chunk_size", [1_000, 50_000, 333_333, LARGE_ROWS])
def test_sorted_frame(large_sorted_df, chunk_size):
    chunks = list(split_by_datetime(large_sorted_df, "dt", chunk_size))
    assert_valid_chunks(large_sorted_df, chunks, "dt", chunk_size)


@pytest.mark.parametrize("chunk_size", [1_000, 50_000, 333_333, LARGE_ROWS])
def test_shuffled_frame(large_shuffled_df, chunk_size):
    chunks = list(split_by_datetime(large_shuffled_df, "dt", chunk_size))
    assert_valid_chunks(large_shuffled_df, chunks, "dt", chunk_size)


@pytest.mark.parametrize("chunk_size", [1_000, 100_000])
def test_sorted_and_shuffled_give_same_chunks(large_sorted_df, large_shuffled_df, chunk_size):
    sorted_chunks = split_by_datetime(large_sorted_df, "dt", chunk_size)
    shuffled_chunks = split_by_datetime(large_shuffled_df, "dt", chunk_size)
    for a, b in zip(sorted_chunks, shuffled_chunks, strict=True):
        assert np.array_equal(keys_of(a["dt"]), keys_of(b["dt"]))
        assert np.array_equal(np.sort(a["id"].to_numpy()), np.sort(b["id"].to_numpy()))


def test_more_than_million_rows_with_many_repeats():
    df = generate_frame(1_500_000, max_repeats=50, seed=1)
    chunks = list(split_by_datetime(df, "dt", 10_000))
    assert_valid_chunks(df, chunks, "dt", 10_000)


class TestMemory:
    """Доп. память на разбиение — мерило требования «экономно расходовать память»."""

    def test_sorted_frame_is_split_without_copying(self, large_sorted_df):
        frame_bytes = large_sorted_df.memory_usage(deep=True).sum()
        peak = peak_memory(lambda: consume(split_by_datetime(large_sorted_df, "dt", 10_000)))
        # Чанки — view, копий данных нет: пик < 1% размера фрейма (~19 MiB).
        assert peak < frame_bytes * 0.01

    def test_shuffled_frame_needs_only_index_arrays(self, large_shuffled_df):
        chunk_size = 10_000
        peak = peak_memory(lambda: consume(split_by_datetime(large_shuffled_df, "dt", chunk_size)))
        # argsort (8 Б/строка) + отсортированные ключи (8 Б/строка) + копия одного чанка;
        # копии всего фрейма (~20 Б/строка сверху) не делается.
        row_bytes = large_shuffled_df.memory_usage(deep=True).sum() / LARGE_ROWS
        assert peak < LARGE_ROWS * 16 * 1.1 + chunk_size * row_bytes * 2

    def test_previous_chunks_are_not_retained(self, large_shuffled_df):
        """Генератор не копит чанки: обойти все ~100 чанков стоит столько же, сколько первый."""
        frame_bytes = large_shuffled_df.memory_usage(deep=True).sum()
        first = peak_memory(lambda: next(split_by_datetime(large_shuffled_df, "dt", 10_000)))
        every = peak_memory(lambda: consume(split_by_datetime(large_shuffled_df, "dt", 10_000)))
        assert every - first < frame_bytes * 0.05
