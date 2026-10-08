"""Демо: генерирует данные на N строк, разбивает их на чанки и печатает статистику.

    python -m chunker --rows 1000000 --chunk-size 100000 [--shuffle] [--output-dir output]
    python -m chunker --rows 1000000000 --chunk-size 1000000 --stream [--batch-size 1000000]
"""

from __future__ import annotations

import argparse
import os
import resource
import shutil
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

from chunker import generate_batches, generate_frame, split_by_datetime, split_stream

MIB = 2**20
GIB = 2**30
MAX_FILES = 1_000
# Пиковый RSS на строку: замер 21–30 Б (отсортировано) и 48–56 Б (перемешано)
# при размере самого фрейма 20 Б/строку; здесь с запасом ~25%.
SORTED_BYTES_PER_ROW = 36
SHUFFLED_BYTES_PER_ROW = 70
# В потоке память зависит не от числа строк, а от размеров пачки и чанка:
# замер 30–53 Б на строку пачки + чанка, здесь с запасом ~25%.
STREAM_BYTES_PER_ROW = 66
CSV_BYTES_PER_ROW = 40  # замер: ~37.6 Б на строку dt,id,value
BASE_BYTES = 150 * MIB  # интерпретатор + numpy/pandas


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    check_resources(args)
    if args.stream:
        run_stream(args)
    else:
        run_in_memory(args)
    print(f"process peak RSS {peak_rss() / MIB:,.0f} MiB")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument(
        "--stream",
        action="store_true",
        help="генерировать и разбивать данные потоком, не держа их целиком в памяти",
    )
    parser.add_argument("--batch-size", type=int, default=1_000_000, help="размер пачки в режиме --stream")
    return parser.parse_args(argv)


def run_in_memory(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    df = generate_frame(args.rows, max_repeats=args.max_repeats, shuffle=args.shuffle, seed=args.seed)
    print(
        f"generated {len(df):,} rows, {count_unique(df['dt']):,} unique dates, "
        f"{df.memory_usage(deep=True).sum() / MIB:.1f} MiB in {time.perf_counter() - started:.2f}s"
    )

    stats = ChunkStats(args.show)
    tracemalloc.start()
    started = time.perf_counter()
    for chunk in split_by_datetime(df, "dt", args.chunk_size):
        stats.add(chunk)
    elapsed = time.perf_counter() - started
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    stats.report(args.chunk_size)
    print(f"split took {elapsed * 1000:.1f} ms, extra peak memory {peak / MIB:.2f} MiB")

    # Отдельным проходом, чтобы запись CSV не искажала замеры выше.
    if args.output_dir and stats.count:
        writer = ChunkWriter(Path(args.output_dir), max_chunks(args))
        started = time.perf_counter()
        for chunk in split_by_datetime(df, "dt", args.chunk_size):
            writer.write(chunk)
        print(f"saved {writer.count:,} chunks to {writer.output_dir}/chunk_*.csv in {time.perf_counter() - started:.2f}s")


def run_stream(args: argparse.Namespace) -> None:
    n_batches = -(-args.rows // args.batch_size)
    print(f"streaming {args.rows:,} rows in {n_batches:,} batches of {args.batch_size:,}")
    writer = ChunkWriter(Path(args.output_dir), max_chunks(args)) if args.output_dir else None

    batches = generate_batches(args.rows, args.batch_size, max_repeats=args.max_repeats, seed=args.seed)
    stats = ChunkStats(args.show, count_unique=True)
    tracemalloc.start()
    started = time.perf_counter()
    for chunk in split_stream(batches, "dt", args.chunk_size):
        stats.add(chunk)
        if writer:
            writer.write(chunk)
        del chunk  # иначе прошлый чанк живёт, пока собирается следующий
    elapsed = time.perf_counter() - started
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    stats.report(args.chunk_size)
    steps = "generation + split" + (" + CSV" if writer else "")
    print(f"{steps} took {elapsed:.2f}s, peak memory {peak / MIB:.1f} MiB")
    if writer and writer.count:
        print(f"saved {writer.count:,} chunks to {writer.output_dir}/chunk_*.csv")


class ChunkStats:
    """Статистика по чанкам на лету: список размеров на миллиарде строк сам съел бы память."""

    def __init__(self, show: int, *, count_unique: bool = False) -> None:
        self.show = show
        self.count_unique = count_unique
        self.count = self.rows = self.unique = self.min = self.max = self.last = 0

    def add(self, chunk: pd.DataFrame) -> None:
        size = len(chunk)
        if self.count < self.show:
            dt = chunk["dt"]
            print(f"  chunk #{self.count}: {size:>9,} rows  [{dt.iloc[0]} .. {dt.iloc[-1]}]")
        self.min = size if self.count == 0 else min(self.min, size)
        self.max = max(self.max, size)
        self.last = size
        self.rows += size
        self.count += 1
        if self.count_unique:
            self.unique += count_unique(chunk["dt"])  # группы не рвутся между чанками — можно складывать

    def report(self, chunk_size: int) -> None:
        if not self.count:
            print("no chunks: frame is empty")
            return
        if self.count > self.show:
            print(f"  ... {self.count - self.show:,} more")
        unique = f", {self.unique:,} unique dates" if self.count_unique else ""
        print(
            f"chunk_size={chunk_size:,}: {self.count:,} chunks, min={self.min:,} max={self.max:,} "
            f"last={self.last:,} total={self.rows:,} rows{unique}"
        )


class ChunkWriter:
    """Пишет чанки в ``output_dir/chunk_NNN.csv``; файлы прошлого запуска удаляются."""

    def __init__(self, output_dir: Path, max_chunks: int) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for old in output_dir.glob("chunk_*.csv"):
            old.unlink()
        self.output_dir = output_dir
        self.width = max(3, len(str(max_chunks - 1)))
        self.count = 0

    def write(self, chunk: pd.DataFrame) -> None:
        chunk.to_csv(self.output_dir / f"chunk_{self.count:0{self.width}d}.csv", index=False)
        self.count += 1


def check_resources(args: argparse.Namespace) -> None:
    """Отказ до генерации данных — вместо OOM-kill контейнера или забитого диска посреди работы."""
    if args.rows < 0 or min(args.chunk_size, args.max_repeats, args.batch_size) < 1:
        raise SystemExit("--rows must be >= 0; --chunk-size, --max-repeats and --batch-size must be >= 1")
    if args.stream and args.shuffle:
        raise SystemExit("--stream needs sorted data and cannot be combined with --shuffle")

    if args.stream:
        needed = BASE_BYTES + min(args.rows, args.batch_size + args.chunk_size) * STREAM_BYTES_PER_ROW
        what = f"batches of {args.batch_size:,} and chunks of {args.chunk_size:,} rows"
        hint = "reduce --batch-size (BATCH_SIZE) or --chunk-size (CHUNK_SIZE)"
    else:
        needed = BASE_BYTES + args.rows * (SHUFFLED_BYTES_PER_ROW if args.shuffle else SORTED_BYTES_PER_ROW)
        what = f"{args.rows:,} rows"
        hint = "reduce --rows (ROWS), use --stream (STREAM=1)"
    available = available_memory()
    if available is not None and needed > available:
        raise SystemExit(
            f"not enough memory: {what} need ~{needed / GIB:.1f} GiB, only {available / GIB:.1f} GiB available; "
            f"{hint} or give Docker more memory (Docker Desktop → Settings → Resources)"
        )

    if args.output_dir:
        if max_chunks(args) > MAX_FILES:
            raise SystemExit(
                f"not saving: up to {max_chunks(args):,} chunks > {MAX_FILES:,} files limit, increase --chunk-size"
            )
        needed_disk = args.rows * CSV_BYTES_PER_ROW
        free = disk_free(Path(args.output_dir))
        if needed_disk > free:
            raise SystemExit(
                f"not enough disk: CSV files need ~{needed_disk / GIB:.1f} GiB, only {free / GIB:.1f} GiB free"
            )


def max_chunks(args: argparse.Namespace) -> int:
    """Все чанки, кроме последнего, не короче chunk_size, поэтому их не больше ceil(rows / chunk_size)."""
    return -(-args.rows // args.chunk_size)


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


def disk_free(path: Path) -> int:
    path = path.resolve()
    while not path.exists():  # папки может ещё не быть — смотрим ближайшую существующую
        path = path.parent
    return shutil.disk_usage(path).free


def peak_rss() -> int:
    """Пиковый RSS процесса; ru_maxrss — в байтах на macOS и в КиБ на Linux."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024


def count_unique(series: pd.Series) -> int:
    """Число уникальных дат без хеш-таблицы nunique (она съедает ~80 Б на уникальное значение)."""
    keys = series.array.view("i8")
    if not series.is_monotonic_increasing:
        keys = np.sort(keys)
    return int(np.count_nonzero(keys[1:] != keys[:-1])) + 1 if len(keys) else 0


if __name__ == "__main__":
    main()
