import argparse
import cProfile
import pickle
import pstats 

from pathlib import Path
from time import perf_counter

from torch import special

from cs336_basics.tokenizers.tokenization import train_bpe
from cs336_basics.tokenizers.config import DATA_PATH_TRAIN, TOKENIZER_SAVE_PATH


def main(): 
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATA_PATH_TRAIN, help="Defaults to DATA_PATH_VALID")
    parser.add_argument("--output_path", type=Path, default=TOKENIZER_SAVE_PATH)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--special_tokens", type=list, default=["<|endoftext|>"],)
    parser.add_argument("--tokenizer_name", type=str, default="TinyStoriesTrain")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--run_profiling", type=bool, default=True)
    args = parser.parse_args()

    start = perf_counter()
    vocab, merge_list = train_bpe(input_path=args.input,
                                  vocab_size=args.vocab_size, 
                                  special_tokens=args.special_tokens, 
                                  num_processes=args.num_workers
                                  )
    elapsed = perf_counter() - start
    print(f"train_bpe finished in {elapsed:.3f}s")
    
    if args.run_profiling: 
        profiler = cProfile.Profile()

        vocab, merge_list = profiler.runcall(
            train_bpe,
            input_path=args.input,
            vocab_size=args.vocab_size,
            special_tokens=args.special_tokens, 
            num_processes=args.num_workers
        )
    output_dir = args.output_path / args.tokenizer_name
    output_dir.mkdir(parents=True, exist_ok=True)

    profiler.dump_stats(output_dir / "stats.prof")
    print(f"Finished: {len(vocab):,} tokens, {len(merge_list):,} merges")
    pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(25)

    with (output_dir / "vocab.pkl").open("wb") as f:
        pickle.dump(vocab, f)
    with (output_dir / "merges.pkl").open("wb") as f:
        pickle.dump(merge_list, f)


if __name__ == "__main__":
    main()
