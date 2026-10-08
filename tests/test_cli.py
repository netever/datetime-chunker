import pandas as pd
import pytest

import chunker.__main__ as cli
from chunker import generate_batches, generate_frame
from chunker.__main__ import main


def read_chunks(output_dir) -> list[pd.DataFrame]:
    return [pd.read_csv(path, parse_dates=["dt"]) for path in sorted(output_dir.glob("chunk_*.csv"))]


def assert_saved_frame(output_dir, expected: pd.DataFrame, chunk_size: int) -> None:
    chunks = read_chunks(output_dir)
    assert all(len(chunk) >= chunk_size for chunk in chunks[:-1])
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev["dt"].iloc[-1] < nxt["dt"].iloc[0]
    restored = pd.concat(chunks, ignore_index=True)
    pd.testing.assert_series_equal(restored["dt"], expected["dt"].reset_index(drop=True), check_dtype=False)
    pd.testing.assert_series_equal(restored["id"], expected["id"].reset_index(drop=True))


@pytest.fixture
def no_generation(monkeypatch):
    """Проверки должны срабатывать до генерации данных."""
    fail = lambda *args, **kwargs: pytest.fail("data must not be generated")  # noqa: E731
    monkeypatch.setattr(cli, "generate_frame", fail)
    monkeypatch.setattr(cli, "generate_batches", fail)


class TestInMemory:
    def test_prints_stats_without_saving_by_default(self, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        main(["--rows", "1000", "--chunk-size", "100"])
        out = capsys.readouterr().out
        assert "generated 1,000 rows" in out
        assert "total=1,000 rows" in out
        assert "process peak RSS" in out
        assert list(tmp_path.iterdir()) == []

    def test_saves_every_chunk_to_csv(self, tmp_path):
        main(["--rows", "1000", "--chunk-size", "100", "--seed", "3", "--output-dir", str(tmp_path)])
        assert len(read_chunks(tmp_path)) >= 9
        assert_saved_frame(tmp_path, generate_frame(1000, seed=3), 100)

    def test_removes_files_from_previous_run(self, tmp_path):
        main(["--rows", "1000", "--chunk-size", "100", "--output-dir", str(tmp_path)])
        main(["--rows", "1000", "--chunk-size", "500", "--output-dir", str(tmp_path)])
        assert sum(len(chunk) for chunk in read_chunks(tmp_path)) == 1000
        assert len(read_chunks(tmp_path)) == 2


class TestStream:
    def test_prints_stats(self, capsys):
        main(["--rows", "100000", "--chunk-size", "10000", "--stream", "--batch-size", "7000"])
        out = capsys.readouterr().out
        assert "streaming 100,000 rows in 15 batches of 7,000" in out
        assert "total=100,000 rows" in out
        unique = pd.concat(generate_batches(100_000, 7_000))["dt"].nunique()
        assert f"{unique:,} unique dates" in out

    def test_saves_every_chunk_to_csv(self, tmp_path):
        main(
            ["--rows", "10000", "--chunk-size", "1000", "--stream", "--batch-size", "777", "--seed", "5",
             "--output-dir", str(tmp_path)]
        )  # fmt: skip
        assert_saved_frame(tmp_path, pd.concat(generate_batches(10_000, 777, seed=5)), 1000)

    def test_memory_check_does_not_depend_on_rows(self, monkeypatch, no_generation):
        """Миллиард строк потоком проходит проверку памяти при 4 GiB."""
        monkeypatch.setattr(cli, "available_memory", lambda: 4 * 2**30)
        cli.check_resources(cli.parse_args(["--rows", "1000000000", "--stream"]))

    def test_huge_chunks_do_not_fit_even_in_stream(self, monkeypatch, no_generation):
        monkeypatch.setattr(cli, "available_memory", lambda: 4 * 2**30)
        with pytest.raises(SystemExit, match="not enough memory: batches of 1,000,000 and chunks of 500,000,000"):
            main(["--rows", "1000000000", "--chunk-size", "500000000", "--stream"])

    def test_cannot_be_combined_with_shuffle(self, no_generation):
        with pytest.raises(SystemExit, match="--shuffle"):
            main(["--stream", "--shuffle"])


class TestResourceChecks:
    @pytest.mark.parametrize("mode", [[], ["--stream"]], ids=["in-memory", "stream"])
    @pytest.mark.parametrize("rows", ["5000", "2000000000"])
    def test_refuses_to_write_too_many_files(self, tmp_path, monkeypatch, no_generation, rows, mode):
        monkeypatch.setattr(cli, "available_memory", lambda: None)
        with pytest.raises(SystemExit, match="files limit"):
            main(["--rows", rows, "--chunk-size", "1", "--output-dir", str(tmp_path), *mode])
        assert list(tmp_path.iterdir()) == []

    def test_refuses_to_start_without_enough_memory(self, monkeypatch, no_generation):
        """2 млрд строк в 4 GiB не влезают: понятная ошибка до генерации, а не OOM-kill."""
        monkeypatch.setattr(cli, "available_memory", lambda: 4 * 2**30)
        with pytest.raises(SystemExit, match="not enough memory: 2,000,000,000 rows.*STREAM=1"):
            main(["--rows", "2000000000"])

    def test_refuses_to_start_without_enough_disk(self, tmp_path, monkeypatch, no_generation):
        monkeypatch.setattr(cli, "disk_free", lambda path: 2**30)
        with pytest.raises(SystemExit, match="not enough disk"):
            main(["--rows", "100000000", "--chunk-size", "1000000", "--stream", "--output-dir", str(tmp_path)])

    def test_runs_when_memory_is_enough(self, monkeypatch, capsys):
        monkeypatch.setattr(cli, "available_memory", lambda: 4 * 2**30)
        main(["--rows", "100000", "--chunk-size", "10000"])
        assert "total=100,000 rows" in capsys.readouterr().out

    def test_available_memory_is_detected(self):
        memory = cli.available_memory()
        assert memory is None or memory > 0

    def test_disk_free_for_missing_dir(self, tmp_path):
        assert cli.disk_free(tmp_path / "not" / "created" / "yet") > 0

    @pytest.mark.parametrize(
        "args",
        [["--rows", "-1"], ["--chunk-size", "0"], ["--max-repeats", "0"], ["--batch-size", "0"]],
        ids=["rows", "chunk", "repeats", "batch"],
    )
    def test_rejects_invalid_arguments(self, args, no_generation):
        with pytest.raises(SystemExit, match="must be"):
            main(args)


@pytest.mark.parametrize("shuffle", [False, True])
def test_count_unique(shuffle):
    df = generate_frame(10_000, shuffle=shuffle)
    assert cli.count_unique(df["dt"]) == df["dt"].nunique()
    assert cli.count_unique(df["dt"].iloc[:0]) == 0
