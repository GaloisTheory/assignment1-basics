# %%
import random
from pathlib import Path

from cs336_basics.tokenizers.tokenizer import Tokenizer

ROOT = next(
    p for p in [Path.cwd(), *Path.cwd().parents] if (p / "artifacts/tokenizers").is_dir()
)
ARTIFACTS = ROOT / "artifacts/tokenizers"
DATA = ROOT / "data"
SPECIAL = "<|endoftext|>"

tinystories_tokenizer = Tokenizer.from_files(
    ARTIFACTS / "TinyStoriesTrain/vocab.pkl",
    ARTIFACTS / "TinyStoriesTrain/merges.pkl",
    special_tokens=[SPECIAL],
)
owt_tokenizer = Tokenizer.from_files(
    ARTIFACTS / "OWTFullRun/vocab.pkl",
    ARTIFACTS / "OWTFullRun/merges.pkl",
    special_tokens=[SPECIAL],
)

print(f"TinyStories vocab: {len(tinystories_tokenizer.vocab):,}")
print(f"OpenWebText vocab: {len(owt_tokenizer.vocab):,}")

# %%
def sample_documents(path, n=100, seed=0):
    """Return n full documents by jumping to random offsets in the file.

    Documents are the text between <|endoftext|> markers. A random offset
    usually lands mid-document, so we skip to the next marker and read until
    the one after that.
    """
    delimiter = SPECIAL.encode("utf-8")
    rng = random.Random(seed)
    documents = []

    with path.open("rb") as f:
        f.seek(0, 2)
        file_size = f.tell()

        while len(documents) < n:
            f.seek(rng.randrange(file_size))

            # Drop the partial document that contains this offset.
            start = None
            while True:
                chunk = f.read(1024 * 1024)
                if chunk == b"":
                    break
                at = chunk.find(delimiter)
                if at != -1:
                    start = f.tell() - len(chunk) + at + len(delimiter)
                    break
            if start is None:
                continue

            # The bytes up to the next marker are one document.
            f.seek(start)
            parts = []
            found_end = False
            while True:
                chunk = f.read(1024 * 1024)
                if chunk == b"":
                    break
                at = chunk.find(delimiter)
                if at != -1:
                    parts.append(chunk[:at])
                    found_end = True
                    break
                parts.append(chunk)
            if not found_end:
                continue

            text = b"".join(parts).strip().decode("utf-8")
            if text:
                documents.append(text)

    return documents


ts_docs = sample_documents(DATA / "TinyStoriesV2-GPT4-train.txt")
owt_docs = sample_documents(DATA / "owt_train.txt")

print("TinyStories documents (chars):", [len(doc) for doc in ts_docs])
print("OpenWebText documents (chars):", [len(doc) for doc in owt_docs])
print()
print("TinyStories preview:", ts_docs[0][:120].replace("\n", " "))
print("OpenWebText preview:", owt_docs[0][:120].replace("\n", " "))

# %%
def compression_ratio(tokenizer, documents):
    """Pooled bytes/token over the sampled documents.

    Bytes are the UTF-8 length of the same text that is encoded.
    """
    total_bytes = 0
    total_tokens = 0
    for doc in documents:
        ids = tokenizer.encode(doc)
        total_bytes += len(doc.encode("utf-8"))
        total_tokens += len(ids)
    return total_bytes / total_tokens, total_bytes, total_tokens


ts_ratio, ts_bytes, ts_tokens = compression_ratio(tinystories_tokenizer, ts_docs)
owt_ratio, owt_bytes, owt_tokens = compression_ratio(owt_tokenizer, owt_docs)

print(f"TinyStories:  {ts_ratio:.3f} bytes/token  ({ts_bytes:,} bytes / {ts_tokens:,} tokens)")
print(f"OpenWebText:  {owt_ratio:.3f} bytes/token  ({owt_bytes:,} bytes / {owt_tokens:,} tokens)")

# %%
ts_ratio_sw, ts_bytes_sw, ts_tokens_sw = compression_ratio(owt_tokenizer, ts_docs)
owt_ratio_sw, owt_bytes_sw, owt_tokens_sw = compression_ratio(tinystories_tokenizer, owt_docs)

print(f"TinyStories:  {ts_ratio_sw:.3f} bytes/token  ({ts_bytes_sw:,} bytes / {ts_tokens:,} tokens)")
print(f"OpenWebText:  {owt_ratio_sw:.3f} bytes/token  ({owt_bytes_sw:,} bytes / {owt_tokens:,} tokens)")


# %%
import math
import time

import matplotlib.pyplot as plt

# First N documents, in file order. The clock below covers encode only.
SIZES = [1, 10, 100, 1_000, 10_000, 100_000]


def read_first_documents(path, n):
    delimiter = SPECIAL.encode("utf-8")
    documents = []
    with path.open("rb") as f:
        buf = b""
        while len(documents) < n:
            chunk = f.read(1024 * 1024)
            if chunk == b"":
                break
            buf += chunk
            parts = buf.split(delimiter)
            buf = parts[-1]
            for part in parts[:-1]:
                text = part.strip().decode("utf-8")
                if text:
                    documents.append(text)
                    if len(documents) == n:
                        return documents
    return documents


def time_encoding(tokenizer, documents, sizes):
    """Cumulative encode time at each document count. Each document is encoded once."""
    rows = []
    elapsed = 0.0
    total_bytes = 0
    next_size = 0
    for i, doc in enumerate(documents, start=1):
        total_bytes += len(doc.encode("utf-8"))
        start = time.perf_counter()
        tokenizer.encode(doc)
        elapsed += time.perf_counter() - start
        if next_size < len(sizes) and i == sizes[next_size]:
            rows.append((i, total_bytes, elapsed))
            print(f"  {i:>7} docs  {total_bytes:>12,} bytes  {elapsed:8.3f} s")
            next_size += 1
    return rows


print("Loading documents...")
ts_scale_docs = read_first_documents(DATA / "TinyStoriesV2-GPT4-train.txt", SIZES[-1])
owt_scale_docs = read_first_documents(DATA / "owt_train.txt", SIZES[-1])
print(f"Loaded {len(ts_scale_docs):,} TinyStories and {len(owt_scale_docs):,} OpenWebText documents")

print("TinyStories")
ts_rows = time_encoding(tinystories_tokenizer, ts_scale_docs, SIZES)
print("OpenWebText")
owt_rows = time_encoding(owt_tokenizer, owt_scale_docs, SIZES)

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

for rows, label in ((ts_rows, "TinyStories"), (owt_rows, "OpenWebText")):
    nbytes = [row[1] for row in rows]
    seconds = [row[2] for row in rows]
    axes[0].plot(nbytes, seconds, marker="o", label=label)
    axes[1].plot([math.log10(b) for b in nbytes], seconds, marker="o", label=label)

axes[0].set_xlabel("bytes")
axes[0].set_ylabel("time (s)")
axes[0].set_title("Encode time vs bytes")
axes[1].set_xlabel("log10(bytes)")
axes[1].set_ylabel("time (s)")
axes[1].set_title("Encode time vs log10(bytes)")
for ax in axes:
    ax.legend()
fig.tight_layout()
fig.savefig(ROOT / "cs336_basics/notebooks/tokenizers/tokenizer_experiments_encode_time.png", dpi=150)
plt.show()

print(f"{'corpus':<14} {'docs':>8} {'bytes':>14} {'seconds':>10} {'bytes/s':>12}")
for name, rows in (("TinyStories", ts_rows), ("OpenWebText", owt_rows)):
    for n, nbytes, seconds in rows:
        print(f"{name:<14} {n:8,} {nbytes:14,} {seconds:10.3f} {nbytes / seconds:12,.0f}")

pile_bytes = 825 * 10**9
for name, rows in (("TinyStories", ts_rows), ("OpenWebText", owt_rows)):
    _, nbytes, seconds = rows[-1]
    hours = pile_bytes / (nbytes / seconds) / 3600
    print(f"Pile (825 GB) at the {name} rate from the largest sample: {hours:.1f} hours")
