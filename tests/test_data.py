import tracemalloc

import numpy as np
import pandas as pd
import pytest

from chunker import generate_batches, generate_frame


@pytest.mark.parametrize("n_rows", [0, 1, 7, 1_000, 1_000_000])
def test_exact_row_count(n_rows):
    assert len(generate_frame(n_rows)) == n_rows


def test_dtypes():
    df = generate_frame(100)
    assert df.dtypes.to_dict() == {
        "dt": np.dtype("datetime64[ns]"),
        "id": np.dtype("int64"),
        "value": np.dtype("float32"),
    }


@pytest.mark.parametrize("max_repeats", [1, 3, 10])
def test_repeats_are_bounded(max_repeats):
    counts = generate_frame(10_000, max_repeats=max_repeats)["dt"].value_counts()
    assert counts.min() >= 1
    assert counts.max() <= max_repeats
    if max_repeats > 1:
        assert counts.max() > 1, "повторения действительно есть"


def test_sorted_by_default():
    df = generate_frame(10_000)
    assert df["dt"].is_monotonic_increasing
    assert np.array_equal(df["id"].to_numpy(), np.arange(10_000))


def test_shuffle_permutes_whole_rows():
    sorted_df = generate_frame(10_000, seed=7)
    shuffled = generate_frame(10_000, seed=7, shuffle=True)
    assert not shuffled["dt"].is_monotonic_increasing
    restored = shuffled.sort_values("id", ignore_index=True)
    pd.testing.assert_frame_equal(restored[["dt", "id"]], sorted_df[["dt", "id"]])


def test_deterministic_with_seed():
    pd.testing.assert_frame_equal(generate_frame(1_000, seed=1), generate_frame(1_000, seed=1))


def test_start_and_step():
    df = generate_frame(1_000, start="2024-05-01 10:00", step="15min", max_repeats=1)
    assert df["dt"].iloc[0] == pd.Timestamp("2024-05-01 10:00")
    assert (df["dt"].diff().dropna() == pd.Timedelta("15min")).all()


@pytest.mark.parametrize(("shuffle", "max_bytes_per_row"), [(False, 26), (True, 38)])
def test_generation_peak_memory_is_close_to_frame_size(shuffle, max_bytes_per_row):
    """Фрейм — 20 Б/строку; генератор не должен держать несколько его копий одновременно."""
    n_rows = 1_000_000
    tracemalloc.start()
    try:
        generate_frame(n_rows, shuffle=shuffle)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak / n_rows < max_bytes_per_row


@pytest.mark.parametrize("kwargs", [{"n_rows": -1}, {"max_repeats": 0}])
def test_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        generate_frame(**kwargs)


class TestGenerateBatches:
    @pytest.mark.parametrize(("n_rows", "batch_size"), [(0, 10), (1, 10), (10, 10), (1_000, 7), (1_000_000, 65_537)])
    def test_batch_sizes_and_total(self, n_rows, batch_size):
        sizes = [len(batch) for batch in generate_batches(n_rows, batch_size)]
        assert sum(sizes) == n_rows
        assert all(size == batch_size for size in sizes[:-1])
        assert all(0 < size <= batch_size for size in sizes)

    def test_batches_form_one_sorted_frame(self):
        df = pd.concat(generate_batches(100_000, 9_999))
        assert df["dt"].is_monotonic_increasing
        assert df.index.equals(pd.RangeIndex(100_000))
        assert np.array_equal(df["id"].to_numpy(), np.arange(100_000))
        assert df.dtypes.to_dict() == generate_frame(10).dtypes.to_dict()

    @pytest.mark.parametrize("batch_size", [1, 3, 1_000])
    def test_repeats_are_bounded_across_batch_boundaries(self, batch_size):
        df = pd.concat(generate_batches(5_000, batch_size, max_repeats=7))
        counts = df["dt"].value_counts()
        assert counts.max() == 7
        assert counts.min() >= 1

    def test_groups_straddle_batch_boundaries(self):
        batches = list(generate_batches(100_000, 1_000, max_repeats=5))
        straddling = sum(a["dt"].iloc[-1] == b["dt"].iloc[0] for a, b in zip(batches, batches[1:]))
        assert straddling > len(batches) // 2

    def test_deterministic_with_seed(self):
        for a, b in zip(generate_batches(10_000, 999, seed=3), generate_batches(10_000, 999, seed=3), strict=True):
            pd.testing.assert_frame_equal(a, b)

    def test_is_lazy_and_memory_bounded_by_batch(self):
        """10M строк генерируются в памяти порядка одной пачки, а не всего объёма."""
        batch_size = 100_000
        tracemalloc.start()
        try:
            for _ in generate_batches(10_000_000, batch_size):
                pass
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert peak < batch_size * 20 * 3

    @pytest.mark.parametrize("kwargs", [{"n_rows": -1}, {"batch_size": 0}, {"max_repeats": 0}])
    def test_invalid_arguments_are_raised_eagerly(self, kwargs):
        with pytest.raises(ValueError):
            generate_batches(**kwargs)
