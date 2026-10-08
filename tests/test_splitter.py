from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest

from chunker import chunk_bounds, split_by_datetime
from tests.helpers import assert_valid_bounds, assert_valid_chunks, keys_of, reference_bounds


def make_df(*timestamps, **columns) -> pd.DataFrame:
    return pd.DataFrame({"dt": pd.to_datetime(list(timestamps)), **columns})


def split_dates(df: pd.DataFrame, chunk_size: int, **kwargs) -> list[list[str]]:
    return [
        chunk["dt"].dt.strftime("%H:%M:%S").tolist()
        for chunk in split_by_datetime(df, "dt", chunk_size, **kwargs)
    ]


class TestTaskExamples:
    @pytest.mark.parametrize("chunk_size", [1, 2])
    def test_small_chunk_size_gives_three_chunks(self, doc_example_df, chunk_size):
        assert split_dates(doc_example_df, chunk_size) == [
            ["00:00:01", "00:00:01"],
            ["00:00:02", "00:00:02", "00:00:02"],
            ["00:00:03"],
        ]

    @pytest.mark.parametrize("chunk_size", [3, 4, 5])
    def test_medium_chunk_size_gives_two_chunks(self, doc_example_df, chunk_size):
        assert split_dates(doc_example_df, chunk_size) == [
            ["00:00:01", "00:00:01", "00:00:02", "00:00:02", "00:00:02"],
            ["00:00:03"],
        ]

    @pytest.mark.parametrize("chunk_size", [6, 7, 1_000_000])
    def test_large_chunk_size_gives_whole_frame(self, doc_example_df, chunk_size):
        chunks = list(split_by_datetime(doc_example_df, "dt", chunk_size))
        assert len(chunks) == 1
        pd.testing.assert_frame_equal(chunks[0], doc_example_df)

    @pytest.mark.parametrize(
        ("chunk_size", "expected_sizes"),
        [(1, [3] * 6), (3, [3] * 6), (4, [6] * 3), (6, [6] * 3), (7, [9, 9]), (13, [15, 3]), (18, [18])],
    )
    def test_frame_from_task_statement(self, chunk_size, expected_sizes):
        dfs = pd.date_range("2023-01-01 00:00:00", "2023-01-01 00:00:05", freq="s")
        df = pd.DataFrame({"dt": dfs.repeat(3)})
        chunks = list(split_by_datetime(df, "dt", chunk_size))
        assert [len(chunk) for chunk in chunks] == expected_sizes
        assert_valid_chunks(df, chunks, "dt", chunk_size)


class TestEdgeCases:
    def test_empty_frame(self):
        df = pd.DataFrame({"dt": pd.Series([], dtype="datetime64[ns]")})
        assert list(split_by_datetime(df, "dt", 3)) == []

    def test_single_row(self):
        df = make_df("2023-01-01")
        chunks = list(split_by_datetime(df, "dt", 5))
        assert len(chunks) == 1
        pd.testing.assert_frame_equal(chunks[0], df)

    @pytest.mark.parametrize("chunk_size", [1, 2, 10, 11])
    def test_all_dates_equal_give_single_chunk(self, chunk_size):
        df = make_df(*["2023-01-01 12:00"] * 10)
        assert [len(chunk) for chunk in split_by_datetime(df, "dt", chunk_size)] == [10]

    def test_all_dates_unique_give_exact_sizes_and_tail(self):
        df = pd.DataFrame({"dt": pd.date_range("2023-01-01", periods=10, freq="min")})
        assert [len(chunk) for chunk in split_by_datetime(df, "dt", 3)] == [3, 3, 3, 1]

    def test_chunk_size_equal_to_len(self):
        df = pd.DataFrame({"dt": pd.date_range("2023-01-01", periods=10, freq="min")})
        assert [len(chunk) for chunk in split_by_datetime(df, "dt", 10)] == [10]

    def test_group_crossing_boundary_is_not_split(self):
        # chunk_size=3 попадает в середину группы из 4 одинаковых дат.
        df = make_df("2023-01-01", "2023-01-02", "2023-01-02", "2023-01-02", "2023-01-02", "2023-01-03")
        chunks = list(split_by_datetime(df, "dt", 3))
        assert [len(chunk) for chunk in chunks] == [5, 1]
        assert_valid_chunks(df, chunks, "dt", 3)

    def test_tail_smaller_than_chunk_size_is_kept(self):
        df = make_df("2023-01-01", "2023-01-01", "2023-01-02")
        assert [len(chunk) for chunk in split_by_datetime(df, "dt", 2)] == [2, 1]


class TestUnsortedInput:
    def test_chunks_are_sorted_and_groups_are_kept_together(self):
        df = make_df(
            "2023-01-01 00:00:03",
            "2023-01-01 00:00:01",
            "2023-01-01 00:00:02",
            "2023-01-01 00:00:01",
            "2023-01-01 00:00:03",
            "2023-01-01 00:00:02",
        )
        assert split_dates(df, 2) == [
            ["00:00:01", "00:00:01"],
            ["00:00:02", "00:00:02"],
            ["00:00:03", "00:00:03"],
        ]
        assert_valid_chunks(df, list(split_by_datetime(df, "dt", 2)), "dt", 2)

    def test_original_order_inside_group_is_preserved(self):
        df = make_df("2023-01-02", "2023-01-01", "2023-01-02", "2023-01-01", tag=["a", "b", "c", "d"])
        assert [chunk["tag"].tolist() for chunk in split_by_datetime(df, "dt", 1)] == [["b", "d"], ["a", "c"]]

    def test_sort_is_stable_on_large_groups(self):
        """На маленьких массивах и нестабильная сортировка ведёт себя стабильно — берём 10k строк."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {"dt": pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 10, size=10_000), unit="s")}
        )
        for chunk in split_by_datetime(df, "dt", 1):
            assert chunk.index.is_monotonic_increasing

    def test_descending_input(self):
        df = pd.DataFrame({"dt": pd.date_range("2023-01-01", periods=9, freq="h")[::-1].repeat(2)})
        chunks = list(split_by_datetime(df, "dt", 5))
        assert [len(chunk) for chunk in chunks] == [6, 6, 6]
        assert_valid_chunks(df, chunks, "dt", 5)


class TestFrameContent:
    def test_other_columns_stay_aligned(self):
        df = make_df("2023-01-02", "2023-01-01", "2023-01-02", value=[1.5, 2.5, 3.5], name=["x", "y", "z"])
        chunks = list(split_by_datetime(df, "dt", 1))
        pd.testing.assert_frame_equal(chunks[0], df.iloc[[1]])
        pd.testing.assert_frame_equal(chunks[1], df.iloc[[0, 2]])

    def test_custom_index_is_preserved(self):
        df = make_df("2023-01-01", "2023-01-01", "2023-01-02", "2023-01-03")
        df.index = ["a", "b", "c", "d"]
        assert [chunk.index.tolist() for chunk in split_by_datetime(df, "dt", 2)] == [["a", "b"], ["c", "d"]]

    def test_tz_aware_column(self):
        df = pd.DataFrame(
            {"dt": pd.date_range("2023-03-26 00:00", periods=6, freq="30min", tz="Europe/Berlin").repeat(2)}
        )
        chunks = list(split_by_datetime(df, "dt", 3))
        assert [len(chunk) for chunk in chunks] == [4, 4, 4]
        assert all(chunk["dt"].dtype == df["dt"].dtype for chunk in chunks)
        assert_valid_chunks(df, chunks, "dt", 3)

    def test_non_nanosecond_unit(self):
        df = pd.DataFrame({"dt": pd.date_range("2023-01-01", periods=4, freq="D").repeat(2).astype("datetime64[s]")})
        chunks = list(split_by_datetime(df, "dt", 3))
        assert [len(chunk) for chunk in chunks] == [4, 4]
        assert_valid_chunks(df, chunks, "dt", 3)

    def test_nat_rows_form_one_group_in_first_chunk(self):
        df = make_df("2023-01-02", None, "2023-01-01", None)
        chunks = list(split_by_datetime(df, "dt", 1))
        assert [len(chunk) for chunk in chunks] == [2, 1, 1]
        assert chunks[0]["dt"].isna().all()

    def test_works_with_any_column_name(self):
        df = pd.DataFrame({0: pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-02"]), "x": [1, 2, 3]})
        assert [len(chunk) for chunk in split_by_datetime(df, 0, 1)] == [2, 1]


class TestMemoryAndLaziness:
    def test_returns_lazy_iterator(self, doc_example_df):
        assert isinstance(split_by_datetime(doc_example_df, "dt", 2), Iterator)

    def test_sorted_input_chunks_are_views(self, doc_example_df):
        df = doc_example_df.assign(value=np.arange(len(doc_example_df), dtype=float))
        for chunk in split_by_datetime(df, "dt", 2):
            assert np.shares_memory(keys_of(chunk["dt"]), keys_of(df["dt"]))
            assert np.shares_memory(chunk["value"].to_numpy(), df["value"].to_numpy())

    def test_assume_sorted_gives_same_result(self, doc_example_df):
        for chunk_size in range(1, 8):
            default = list(split_by_datetime(doc_example_df, "dt", chunk_size))
            trusted = list(split_by_datetime(doc_example_df, "dt", chunk_size, assume_sorted=True))
            assert len(default) == len(trusted)
            for a, b in zip(default, trusted):
                pd.testing.assert_frame_equal(a, b)

    def test_input_frame_is_not_modified(self):
        df = make_df("2023-01-03", "2023-01-01", "2023-01-02", "2023-01-01")
        before = df.copy()
        list(split_by_datetime(df, "dt", 2))
        pd.testing.assert_frame_equal(df, before)


class TestValidation:
    @pytest.mark.parametrize("chunk_size", [0, -1])
    def test_non_positive_chunk_size(self, doc_example_df, chunk_size):
        with pytest.raises(ValueError, match="chunk_size"):
            split_by_datetime(doc_example_df, "dt", chunk_size)

    @pytest.mark.parametrize("chunk_size", [1.5, "3", None, True])
    def test_non_integer_chunk_size(self, doc_example_df, chunk_size):
        with pytest.raises(TypeError, match="chunk_size"):
            split_by_datetime(doc_example_df, "dt", chunk_size)

    def test_numpy_integer_chunk_size_is_accepted(self, doc_example_df):
        assert len(list(split_by_datetime(doc_example_df, "dt", np.int64(3)))) == 2

    def test_missing_column(self, doc_example_df):
        with pytest.raises(KeyError):
            split_by_datetime(doc_example_df, "missing", 2)

    @pytest.mark.parametrize(
        "values",
        [[1, 2, 3], ["2023-01-01", "2023-01-02", "2023-01-03"], pd.to_datetime(["2023-01-01"] * 3).astype(object)],
        ids=["int", "str", "object-datetime"],
    )
    def test_non_datetime_column(self, values):
        with pytest.raises(TypeError, match="datetime64"):
            split_by_datetime(pd.DataFrame({"dt": values}), "dt", 2)

    def test_errors_are_raised_eagerly(self, doc_example_df):
        """Ошибка — сразу при вызове, а не при первой итерации генератора."""
        with pytest.raises(ValueError):
            split_by_datetime(doc_example_df, "dt", 0)
        with pytest.raises(ValueError):
            chunk_bounds(np.arange(3), 0)


class TestChunkBounds:
    def test_empty(self):
        assert list(chunk_bounds(np.array([], dtype=np.int64), 3)) == []

    def test_example(self):
        keys = np.array([1, 1, 2, 2, 2, 3])
        assert list(chunk_bounds(keys, 2)) == [(0, 2), (2, 5), (5, 6)]
        assert list(chunk_bounds(keys, 3)) == [(0, 5), (5, 6)]
        assert list(chunk_bounds(keys, 6)) == [(0, 6)]


@pytest.mark.parametrize("seed", range(30))
def test_matches_naive_reference_on_random_data(seed):
    """Сверка с наивным эталоном на случайных маленьких фреймах (в т.ч. перемешанных)."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(0, 300))
    df = pd.DataFrame(
        {
            "dt": pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, max(n // 3, 1), size=n), unit="s"),
            "id": np.arange(n),
        }
    )
    if seed % 2:
        df = df.sort_values("dt", kind="stable")
    chunk_size = int(rng.integers(1, n + 3))

    sorted_keys = np.sort(keys_of(df["dt"]), kind="stable")
    expected = reference_bounds(sorted_keys, chunk_size)
    actual = list(chunk_bounds(sorted_keys, chunk_size))
    assert actual == expected
    assert_valid_bounds(sorted_keys, actual, chunk_size)

    chunks = list(split_by_datetime(df, "dt", chunk_size))
    assert [len(chunk) for chunk in chunks] == [stop - start for start, stop in expected]
    assert_valid_chunks(df, chunks, "dt", chunk_size)
