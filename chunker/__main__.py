"""Демо: генерирует фрейм на N строк, разбивает его на чанки и печатает статистику.

    python -m chunker --rows 1000000 --chunk-size 100000 [--shuffle] [--output-dir output]
"""

from __future__ import annotations

import argparse
import os
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

from chunker import generate_frame, split_by_datetime

MIB = 2**20
GIB = 2**30
MAX_FILES = 1_000
# Пиковый RSS на строку: замер 21–30 Б (отсортировано) и 48–56 Б (перемешано)
# при размере самого фрейма 20 Б/строку; здесь с запасом ~25%.
SORTED_BYTES_PER_ROW = 36
SHUFFLED_BYTES_PER_ROW = 70
BASE_BYTES = 150 * MIB  # интерпретатор + numpy/pandas


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="chunker", description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, default=1_000_000, help="число строк (по умолчанию 1 000 000)")
    parser.add_argument("--chunk-size", type=int, default=100_000, help="желаемый размер чанка")
    parser.add_argument("--max-repeats", type=int, default=5, help="максимум повторов одной даты")
    parser.add_argument("--shuffle", action="store_true", help="перемешать строки (несортированный вход)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show", type=int, default=5, help="сколько первых чанков вывести")
    parser.add_argument(
        "--output-dir",
        default="",
        help="сохранить каждый чанк в <dir>/chunk_NNN.csv (по умолчанию выключено)",
    )
    args = parser.parse_args(argv)

    check_resources(args)

    started = time.perf_counter()
    df = generate_frame(args.rows, max_repeats=args.max_repeats, shuffle=args.shuffle, seed=args.seed)
    print(
        f"generated {len(df):,} rows, {count_unique(df['dt']):,} unique dates, "
        f"{df.memory_usage(deep=True).sum() / MIB:.1f} MiB in {time.perf_counter() - started:.2f}s"
    )

    sizes: list[int] = []
    tracemalloc.start()
    started = time.perf_counter()
    for chunk in split_by_datetime(df, "dt", args.chunk_size):
        if len(sizes) < args.show:
            dt = chunk["dt"]
            print(f"  chunk #{len(sizes)}: {len(chunk):>9,} rows  [{dt.iloc[0]} .. {dt.iloc[-1]}]")
        sizes.append(len(chunk))
    elapsed = time.perf_counter() - started
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    if not sizes:
        print("no chunks: frame is empty")
        return
    if len(sizes) > args.show:
        print(f"  ... {len(sizes) - args.show:,} more")
    print(
        f"chunk_size={args.chunk_size:,}: {len(sizes):,} chunks, "
        f"min={min(sizes):,} max={max(sizes):,} last={sizes[-1]:,} total={sum(sizes):,} rows"
    )
    print(f"split took {elapsed * 1000:.1f} ms, extra peak memory {peak / MIB:.2f} MiB")

    # Отдельным проходом, чтобы запись CSV не искажала замеры выше.
    if args.output_dir:
        save_chunks(df, args.chunk_size, Path(args.output_dir), len(sizes))


def check_resources(args: argparse.Namespace) -> None:
    """Отказ до генерации данных, а не OOM-kill контейнера посреди работы."""
    if args.rows < 0 or args.chunk_size < 1 or args.max_repeats < 1:
        raise SystemExit("--rows must be >= 0, --chunk-size and --max-repeats must be >= 1")

    needed = BASE_BYTES + args.rows * (SHUFFLED_BYTES_PER_ROW if args.shuffle else SORTED_BYTES_PER_ROW)
    available = available_memory()
    if available is not None and needed > available:
        raise SystemExit(
            f"not enough memory: {args.rows:,} rows need ~{needed / GIB:.1f} GiB, "
            f"only {available / GIB:.1f} GiB available; reduce --rows (ROWS) "
            "or give Docker more memory (Docker Desktop → Settings → Resources)"
        )

    # Чанк не длиннее chunk_size + max_repeats - 1 строк, отсюда нижняя оценка числа файлов.
    min_chunks = -(-args.rows // (args.chunk_size + args.max_repeats - 1))
    if args.output_dir and min_chunks > MAX_FILES:
        raise SystemExit(f"not saving: {min_chunks:,}+ chunks > {MAX_FILES:,} files limit, increase --chunk-size")


def available_memory() -> int | None:
    """Доступная память: минимум из лимита cgroup контейнера и MemAvailable; вне Linux — объём RAM."""
    limits = []
    try:
        limit = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if limit != "max":
            limits.append(int(limit))
    except (OSError, ValueError):
        pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                limits.append(int(line.split()[1]) * 1024)
    except (OSError, ValueError):
        pass
    if not limits:
        try:
            limits.append(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        except (AttributeError, OSError, ValueError):
            return None
    return min(limits)


def count_unique(series: pd.Series) -> int:
    """Число уникальных дат без хеш-таблицы nunique (она съедает ~80 Б на уникальное значение)."""
    keys = series.array.view("i8")
    if not series.is_monotonic_increasing:
        keys = np.sort(keys)
    return int(np.count_nonzero(keys[1:] != keys[:-1])) + 1 if len(keys) else 0


def save_chunks(df: pd.DataFrame, chunk_size: int, output_dir: Path, n_chunks: int) -> None:
    """Пишет каждый чанк в ``output_dir/chunk_NNN.csv``; файлы прошлого запуска удаляются."""
    if n_chunks > MAX_FILES:
        raise SystemExit(f"not saving: {n_chunks:,} chunks > {MAX_FILES:,} files limit, increase --chunk-size")
    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob("chunk_*.csv"):
        old.unlink()

    width = max(3, len(str(n_chunks - 1)))
    started = time.perf_counter()
    for i, chunk in enumerate(split_by_datetime(df, "dt", chunk_size)):
        chunk.to_csv(output_dir / f"chunk_{i:0{width}d}.csv", index=False)
    print(f"saved {n_chunks:,} chunks to {output_dir}/chunk_*.csv in {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
