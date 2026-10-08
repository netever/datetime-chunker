"""Потоковое разбиение: split_stream должен давать ровно то же, что split_by_datetime на склеенных пачках."""

import tracemalloc
from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest

from chunker import generate_batches, split_by_datetime, split_stream
from tests.conftest import LARGE_ROWS
from tests.helpers import assert_valid_chunks, keys_of


def batches_of(df: pd.DataFrame, sizes) -> list[pd.DataFrame]:
    """Режет фрейм на идущие подряд пачки заданных размеров (в т.ч. пустые)."""
    assert sum(sizes) == len(df)
    bounds = np.cumsum([0, *sizes])
    return [df.iloc[start:stop] for start, stop in zip(bounds, bounds[1:])]


def assert_same_chunks(actual, expected) -> None:
    actual, expected = list(actual), list(expected)
    assert [len(chunk) for chunk in actual] == [len(chunk) for chunk in expected]
    for a, e in zip(actual, expected):
        pd.testing.assert_frame_equal(a, e)


def dates_df(*timestamps) -> pd.DataFrame:
    return pd.DataFrame({"dt": pd.to_datetime(list(timestamps)), "id": np.arange(len(timestamps))})


@pytest.mark.parametrize("chunk_size", range(1, 8))
@pytest.mark.parametrize(
    "sizes", [[6], [1, 5], [2, 4], [3, 3], [5, 1], [1] * 6, [0, 2, 0, 4, 0]], ids=lambda s: "-".join(map(str, s))
)
def test_task_example_with_any_batching(doc_example_df, chunk_size, sizes):
    assert_same_chunks(
        split_stream(batches_of(doc_example_df, sizes), "dt", chunk_size),
        split_by_datetime(doc_example_df, "dt", chunk_size),
    )


class TestEdgeCases:
    def test_returns_lazy_iterator(self, doc_example_df):
        assert isinstance(split_stream([doc_example_df], "dt", 2), Iterator)

    def test_empty_stream(self):
        assert list(split_stream([], "dt", 3)) == []

    def test_only_empty_batches(self, doc_example_df):
        assert list(split_stream([doc_example_df.iloc[:0]] * 3, "dt", 3)) == []

    @pytest.mark.parametrize("chunk_size", [1, 3, 10, 100])
    def test_group_spanning_many_batches_stays_whole(self, chunk_size):
        df = dates_df(*["2023-01-01"] * 10, "2023-01-02")
        chunks = list(split_stream(batches_of(df, [2, 2, 2, 2, 2, 1]), "dt", chunk_size))
        assert len(chunks[0]) >= 10
        assert chunks[0]["dt"].iloc[:10].nunique() == 1

    def test_tail_is_kept(self):
        df = dates_df("2023-01-01", "2023-01-01", "2023-01-02")
        assert [len(chunk) for chunk in split_stream(batches_of(df, [1, 1, 1]), "dt", 2)] == [2, 1]

    def test_chunk_inside_one_batch_is_a_view(self):
        df = pd.DataFrame({"dt": pd.date_range("2023-01-01", periods=100, freq="s"), "value": np.arange(100.0)})
        batches = batches_of(df, [50, 50])
        first = next(split_stream(batches, "dt", 10))
        assert np.shares_memory(first["value"].to_numpy(), batches[0]["value"].to_numpy())

    def test_chunk_across_batches_is_glued(self):
        df = dates_df("2023-01-01", "2023-01-02", "2023-01-02", "2023-01-02", "2023-01-03")
        chunks = list(split_stream(batches_of(df, [2, 1, 2]), "dt", 2))
        assert [chunk["id"].tolist() for chunk in chunks] == [[0, 1, 2, 3], [4]]
        assert chunks[0].index.tolist() == [0, 1, 2, 3]

    def test_tz_aware_column(self):
        df = pd.DataFrame({"dt": pd.date_range("2023-01-01", periods=6, freq="h", tz="Asia/Tokyo").repeat(3)})
        assert_same_chunks(split_stream(batches_of(df, [4, 4, 4, 6]), "dt", 5), split_by_datetime(df, "dt", 5))

    def test_works_with_read_csv_chunks(self, tmp_path):
        """Типичный источник потока — pd.read_csv(..., chunksize=N)."""
        path = tmp_path / "data.csv"
        next(generate_batches(10_000, 10_000, max_repeats=7, seed=1)).to_csv(path, index=False)
        whole = pd.read_csv(path, parse_dates=["dt"])
        with pd.read_csv(path, parse_dates=["dt"], chunksize=777) as reader:
            assert_same_chunks(split_stream(reader, "dt", 1_000), split_by_datetime(whole, "dt", 1_000))


class TestValidation:
    def test_unsorted_batch(self):
        df = dates_df("2023-01-02", "2023-01-01")
        with pytest.raises(ValueError, match="sorted"):
            list(split_stream([df], "dt", 1))

    def test_batches_out_of_order(self):
        df = dates_df("2023-01-01", "2023-01-02", "2023-01-03")
        with pytest.raises(ValueError, match="sorted"):
            list(split_stream([df.iloc[2:], df.iloc[:2]], "dt", 1))

    @pytest.mark.parametrize(("chunk_size", "error"), [(0, ValueError), (1.5, TypeError)])
    def test_invalid_chunk_size_is_raised_eagerly(self, doc_example_df, chunk_size, error):
        with pytest.raises(error, match="chunk_size"):
            split_stream([doc_example_df], "dt", chunk_size)

    def test_non_datetime_column(self):
        with pytest.raises(TypeError, match="datetime64"):
            list(split_stream([pd.DataFrame({"dt": [1, 2, 3]})], "dt", 2))


@pytest.mark.parametrize("seed", range(40))
def test_matches_in_memory_split_on_random_batching(seed):
    """Случайные данные, случайная нарезка на пачки (пустые, по одной строке, рвущие группы)."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(0, 400))
    offsets = np.sort(rng.integers(0, max(n // 4, 1), size=n))
    df = pd.DataFrame({"dt": pd.Timestamp("2023-01-01") + pd.to_timedelta(offsets, unit="s"), "id": np.arange(n)})
    cuts = np.sort(rng.integers(0, n + 1, size=int(rng.integers(0, 20))))
    sizes = np.diff(np.concatenate(([0], cuts, [n])))
    chunk_size = int(rng.integers(1, n + 3))
    assert_same_chunks(
        split_stream(batches_of(df, sizes), "dt", chunk_size),
        split_by_datetime(df, "dt", chunk_size),
    )


class TestLarge:
    @pytest.mark.parametrize("chunk_size", [1_000, 65_536, 333_333, 2 * LARGE_ROWS])
    def test_million_rows_match_in_memory_split(self, chunk_size):
        # Размер пачки не кратен размеру чанка: группы и чанки постоянно попадают на стыки.
        batches = list(generate_batches(LARGE_ROWS, 65_537, max_repeats=7, seed=11))
        whole = pd.concat(batches)
        streamed = list(split_stream(batches, "dt", chunk_size))
        assert_valid_chunks(whole, streamed, "dt", chunk_size)
        expected = list(split_by_datetime(whole, "dt", chunk_size))
        assert [len(chunk) for chunk in streamed] == [len(chunk) for chunk in expected]
        for a, e in zip(streamed, expected):
            assert np.array_equal(keys_of(a["dt"]), keys_of(e["dt"]))
            assert np.array_equal(a["id"].to_numpy(), e["id"].to_numpy())

    def test_memory_does_not_depend_on_number_of_rows(self):
        """10M строк (~190 MiB данных) проходят через поток в памяти размером с пачку и чанк."""

        def peak(n_rows: int) -> int:
            tracemalloc.start()
            try:
                for _ in split_stream(generate_batches(n_rows, 100_000), "dt", 100_000):
                    pass
                return tracemalloc.get_traced_memory()[1]
            finally:
                tracemalloc.stop()

        small, large = peak(1_000_000), peak(10_000_000)
        assert large < small * 1.5
        assert large < 20 * 2**20
