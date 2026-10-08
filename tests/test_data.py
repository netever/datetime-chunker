import tracemalloc

import numpy as np
import pandas as pd
import pytest

from chunker import generate_frame


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
