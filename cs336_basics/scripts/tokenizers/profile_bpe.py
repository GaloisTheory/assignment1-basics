"""Compare serial BPE training with process pools; profile the parent process.

Run from the assignment directory: uv run python cs336_basics/scripts/tokenizers/profile_bpe.py
Worker internals are not captured by the parent's cProfile instance.
"""

import argparse
import cProfile
import json
import pstats
from datetime import datetime
from functools import partial
from pathlib import Path
from statistics import median
from time import perf_counter
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Defaults to DATA_PATH_VALID")
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--repeats", type=int, default=1,
                        help="Unprofiled runs per configuration (default: 1)")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("profiles"))
    parser.add_argument("--timing-only", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or args.vocab_size < 257 or args.top < 1:
        parser.error("repeats/top must be positive and vocab-size must be >= 257")

    # Import only in the parent entry point; spawn can safely load this script.
    from cs336_basics.tokenizers import tokenization
    from cs336_basics.tokenizers.config import DATA_PATH_VALID

    input_path = (args.input or Path(DATA_PATH_VALID)).resolve()
    if not input_path.is_file():
        parser.error(f"Input file does not exist: {input_path}")
    output = args.output / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True)
    serial_pretokenize = tokenization.pretokenize
    configurations = [("serial", None)] + [(f"pool-{n}", n) for n in (1, 2, 4, 8)]

    def run(workers):
        # train_bpe's num_processes=1 normally selects serial pre-tokenization.
        # Temporarily route that call to the existing chunked implementation
        # to measure a real one-worker pool as well. No trainer source changes.
        pretokenizer = serial_pretokenize if workers is None else partial(
            tokenization.pretokenize_chunked, num_processes=workers
        )
        with patch.object(tokenization, "pretokenize", pretokenizer):
            return tokenization.train_bpe(
                input_path=str(input_path), vocab_size=args.vocab_size,
                special_tokens=["<|endoftext|>"], num_processes=1,
            )

    report = {
        "input": str(input_path), "vocab_size": args.vocab_size,
        "repeats": args.repeats, "profile_scope": "parent process only",
        "runs": [],
    }
    reference = None
    print(f"Input: {input_path}\nVocabulary: {args.vocab_size:,}\nOutput: {output}", flush=True)
    print("Timing runs include pool startup, reading, pre-tokenization and merging.", flush=True)
    for label, workers in configurations:
        times = []
        for repeat in range(args.repeats):
            print(f"Timing {label}, run {repeat + 1}/{args.repeats}...", flush=True)
            start = perf_counter()
            result = run(workers)
            elapsed = perf_counter() - start
            if reference is None:
                reference = result
            if result != reference:
                raise RuntimeError(f"{label}: vocabulary or ordered merges differ from serial")
            times.append(elapsed)
            print(f"  {elapsed:.3f}s; matches serial", flush=True)
        report["runs"].append({"configuration": label, "seconds": times,
                               "median_seconds": median(times)})

    baseline = report["runs"][0]["median_seconds"]
    print("\nUnprofiled elapsed time (median)")
    print(f"{'Configuration':<16} {'Seconds':>10} {'Speedup':>10}")
    for row in report["runs"]:
        row["speedup"] = baseline / row["median_seconds"]
        print(f"{row['configuration']:<16} {row['median_seconds']:>10.3f} {row['speedup']:>9.2f}x")
    (output / "timings.json").write_text(json.dumps(report, indent=2) + "\n")

    if not args.timing_only:
        print("\nSeparate profiling runs: parent only; worker function details are absent.", flush=True)
        for label, workers in configurations:
            print(f"Profiling {label}...", flush=True)
            profiler = cProfile.Profile()
            result = profiler.runcall(run, workers)
            if result != reference:
                raise RuntimeError(f"{label}: profiled result differs from serial")
            profiler.dump_stats(str(output / f"{label}.prof"))
            with (output / f"{label}.txt").open("w") as stream:
                pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats(
                    "cumulative"
                ).print_stats(args.top)
            pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(args.top)
    print(f"\nSaved results to {output.resolve()}")


if __name__ == "__main__":
    main()
