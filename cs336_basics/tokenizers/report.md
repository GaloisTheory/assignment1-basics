# BPE tokenizer training report

Date: 2026-09-21

## Setup

TinyStories byte-level BPE with a vocabulary of 10,000 tokens, including
`<|endoftext|>`, and 9,743 learned merges. Hardware: Apple M3 MacBook Air,
8 CPU cores (4 performance and 4 efficiency), 24 GiB memory.
Results below are user-supplied runs, not repeated benchmark averages.

## Algorithm improvement

The baseline recounted all pairs and reconstructed all pretokens after every
merge. Maintaining a pair-to-pretoken index and incrementally updating counts
removed much of that repeated work. On the validation corpus, profiled merge
time fell from 131.129 s to 6.585 s (about 20x), and total profiled time fell
from 138.788 s to 14.321 s (about 9.7x). These are profiled timings; the reduction
in profiler overhead contributes to the measured improvement.

## Multiprocessing

Pre-tokenization uses independent chunks aligned to special-token boundaries.
Each worker returns pretoken frequencies, which are summed in the parent.

Validation-corpus timings without cProfile:

| Execution | Elapsed time | Speedup over serial |
|---|---:|---:|
| Serial | 9.628 s | 1.00x |
| 1 worker | 9.836 s | 0.98x |
| 2 workers | 8.405 s | 1.15x |
| 4 workers | 7.214 s | 1.33x |
| 8 workers | 7.072 s | 1.36x |

All configurations returned identical vocabularies and ordered merges in this
comparison. Four workers captured most of the validation-run benefit; increasing
to eight saved only 0.142 s in these single measurements.

Full TinyStories training timings without cProfile:

| Workers | Run | Training time |
|---|---|---:|
| 4 | Earlier | 138.628 s |
| 8 | Earlier | 107.512 s |
| 8 | Latest, profiling worker-count bug fixed | 111.034 s |

Eight workers reduced elapsed time by 22.4% in the earlier run and 19.9% in the
latest run (1.29x and 1.25x speedup over the four-worker measurement), reaching
about 1 minute 48 seconds and 1 minute 51 seconds respectively. Both are under
two minutes. The larger corpus benefited more from the extra workers than the
validation corpus. These times measure the training call, excluding subsequent
profiling and artifact serialization; they are individual runs, not averages.

## Profiling interpretation

Full-training profiles with known worker counts:

| Workers | Total | Pre-tokenization | Merging |
|---|---:|---:|---:|
| 4 | 141.706 s | 120.271 s | 21.423 s |
| 8 | 117.304 s | 95.218 s | 22.072 s |

With eight workers, pre-tokenization accounts for 81.2% of profiled elapsed time,
and merging for 18.8%. Compared with the four-worker profile, pre-tokenization
takes 20.8% less time and total time falls 17.2%; merging is roughly unchanged.
Pre-tokenization remains the dominant phase on the full training corpus, and
the corrected eight-worker profile also finishes under two minutes.
Parent-process wait/poll entries reflect waiting for worker results and pool
coordination; they do not establish that lock operations themselves are the
computational bottleneck. Worker internals are not profiled by this profiler.

Resolved measurement issue: the earlier `train_tokenizer.py` omitted
`num_processes` from `profiler.runcall`, so its profile after the eight-worker
timing run actually used the default four workers. That 142.563 s total
(121.432 s pre-tokenization, 21.120 s merging) must not be treated as an
eight-worker profile. The script now forwards `args.num_workers` to both calls,
and the latest eight-worker profile above demonstrates the improvement.
These measurements do not establish that profiling substantially reduces the
benefit of extra workers: they are separate single runs, and worker internals
are outside the parent's profiler.

## Vocabulary inspection

Direct inspection of the locally generated full-training artifact
`artifacts/tokenizers/TinyStoriesTrain_tokenizer.pkl` establishes:

- 10,000 vocabulary entries and 9,743 ordered merges.
- Entry lengths from 1 to 15 bytes; mean length 5.7913 bytes.
- Most common length: 5 bytes (1,835 entries).
- Three entries tie for longest at 15 bytes: `b' responsibility'`,
  `b' disappointment'`, and `b' accomplishment'` (including leading spaces).

The notebook `cs336_basics/notebooks/tokenizers/vocab_length_distribution.ipynb`
still loads `TinyStoriesValid_tokenizer.pkl` in the saved version inspected on
2026-09-21, not the full-training tokenizer. The user reports checking the
training vocabulary, but that change/result is not present in this saved file;
the full-training observations above were obtained directly from the artifact.
The saved notebook outputs establish the following validation-vocabulary
observations:

- 10,000 vocabulary entries, with lengths from 1 to 15 bytes.
- Mean vocabulary-entry length: 5.76 bytes; most common length: 5 bytes
  (1,842 entries).
- Longest entry: `b' accomplishment'`, 15 bytes including the leading space.
- Other long entries include `b' understanding'`, `b' uncomfortable'`, and
  `b' neighbourhood'`.

Leading spaces are consistent with the pre-tokenization regex. Whole words in
the learned vocabulary are consistent with repeated within-pretoken BPE merges.
Lengths are byte counts, not necessarily Unicode character counts. The histogram
weights each vocabulary entry once, includes the special token, and does not
measure token-use frequencies or corpus compression. In particular, 5.76 bytes
per vocabulary entry is not the corpus's bytes-per-token compression ratio.

## Remaining observations and next step

Peak training memory was not recorded; this is the remaining missing
measurement for the TinyStories exercise's time/memory discussion. The
full-training vocabulary's longest tokens have now been checked directly.
No additional optimization is needed to move on.

Defer `train_bpe_expts_owt` and proceed to section 2.6 (encoding and decoding).
This is a learning-priority decision, not completion of the OpenWebText exercise.
The deferred exercise explores how a broader training distribution and a 32K
vocabulary change learned tokens. Later cross-domain compression comparisons
require encoding; a small OpenWebText sample encoded with the TinyStories
tokenizer can provide a cheap first observation, but does not replace training
and comparing both tokenizers. Dataset and vocabulary-size changes should be
distinguished when interpreting any comparison.
