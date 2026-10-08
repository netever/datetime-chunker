"""Демо: генерирует фрейм на N строк, разбивает его на чанки и печатает статистику.

    python -m chunker --rows 1000000 --chunk-size 100000 [--shuffle]
"""

from __future__ import annotations

import argparse
import time
import tracemalloc

from chunker import generate_frame, split_by_datetime

MIB = 2**20


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="chunker", description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, default=1_000_000, help="число строк (по умолчанию 1 000 000)")
    parser.add_argument("--chunk-size", type=int, default=100_000, help="желаемый размер чанка")
    parser.add_argument("--max-repeats", type=int, default=5, help="максимум повторов одной даты")
    parser.add_argument("--shuffle", action="store_true", help="перемешать строки (несортированный вход)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show", type=int, default=5, help="сколько первых чанков вывести")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    df = generate_frame(args.rows, max_repeats=args.max_repeats, shuffle=args.shuffle, seed=args.seed)
    print(
        f"generated {len(df):,} rows, {df['dt'].nunique():,} unique dates, "
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


if __name__ == "__main__":
    main()
