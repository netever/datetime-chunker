import pandas as pd
import pytest

import chunker.__main__ as cli
from chunker import generate_frame
from chunker.__main__ import main


def read_chunks(output_dir) -> list[pd.DataFrame]:
    return [pd.read_csv(path, parse_dates=["dt"]) for path in sorted(output_dir.glob("chunk_*.csv"))]


def test_prints_stats_without_saving_by_default(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    main(["--rows", "1000", "--chunk-size", "100"])
    out = capsys.readouterr().out
    assert "generated 1,000 rows" in out
    assert "total=1,000 rows" in out
    assert list(tmp_path.iterdir()) == []


def test_saves_every_chunk_to_csv(tmp_path):
    main(["--rows", "1000", "--chunk-size", "100", "--seed", "3", "--output-dir", str(tmp_path)])
    chunks = read_chunks(tmp_path)
    assert len(chunks) >= 9
    assert all(len(chunk) >= 100 for chunk in chunks[:-1])
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev["dt"].iloc[-1] < nxt["dt"].iloc[0]

    expected = generate_frame(1000, seed=3)
    restored = pd.concat(chunks, ignore_index=True)
    pd.testing.assert_series_equal(restored["dt"], expected["dt"], check_dtype=False)
    pd.testing.assert_series_equal(restored["id"], expected["id"])


def test_removes_files_from_previous_run(tmp_path):
    main(["--rows", "1000", "--chunk-size", "100", "--output-dir", str(tmp_path)])
    main(["--rows", "1000", "--chunk-size", "500", "--output-dir", str(tmp_path)])
    assert sum(len(chunk) for chunk in read_chunks(tmp_path)) == 1000
    assert len(read_chunks(tmp_path)) == 2


@pytest.mark.parametrize("rows", ["5000", "2000000000"], ids=["after-split", "before-generation"])
def test_refuses_to_write_too_many_files(tmp_path, monkeypatch, rows):
    monkeypatch.setattr(cli, "available_memory", lambda: None)
    with pytest.raises(SystemExit, match="files limit"):
        main(["--rows", rows, "--chunk-size", "1", "--output-dir", str(tmp_path)])
    assert list(tmp_path.iterdir()) == []


def test_refuses_to_start_without_enough_memory(monkeypatch, capsys):
    """2 млрд строк в 4 GiB не влезают: понятная ошибка до генерации, а не OOM-kill."""
    monkeypatch.setattr(cli, "available_memory", lambda: 4 * 2**30)
    monkeypatch.setattr(cli, "generate_frame", lambda *a, **kw: pytest.fail("data must not be generated"))
    with pytest.raises(SystemExit, match="not enough memory: 2,000,000,000 rows"):
        main(["--rows", "2000000000"])


def test_runs_when_memory_is_enough(monkeypatch, capsys):
    monkeypatch.setattr(cli, "available_memory", lambda: 4 * 2**30)
    main(["--rows", "100000", "--chunk-size", "10000"])
    assert "total=100,000 rows" in capsys.readouterr().out


def test_available_memory_is_detected():
    memory = cli.available_memory()
    assert memory is None or memory > 0


@pytest.mark.parametrize(
    "args", [["--rows", "-1"], ["--chunk-size", "0"], ["--max-repeats", "0"]], ids=["rows", "chunk", "repeats"]
)
def test_rejects_invalid_arguments(args):
    with pytest.raises(SystemExit, match="must be"):
        main(args)


@pytest.mark.parametrize("shuffle", [False, True])
def test_count_unique(shuffle):
    df = generate_frame(10_000, shuffle=shuffle)
    assert cli.count_unique(df["dt"]) == df["dt"].nunique()
    assert cli.count_unique(df["dt"].iloc[:0]) == 0
