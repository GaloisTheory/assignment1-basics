"""Save full datasets using the existing Tokenizer, with bounded document buffering.

Run from the assignment root with:
    .venv/bin/python -m cs336_basics.scripts.tokenizers.run_encoding --corpus tinystories
    .venv/bin/python -m cs336_basics.scripts.tokenizers.run_encoding --corpus owt
"""

import argparse
import hashlib
import json
import time
from datetime import datetime, UTC
from pathlib import Path

import numpy as np

from cs336_basics.tokenizers.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parents[3]
SPECIAL = "<|endoftext|>"
SOURCES = {
    "tinystories": ("TinyStoriesTrain", "TinyStoriesV2-GPT4-valid.txt", "TinyStoriesV2-GPT4-train.txt"),
    "owt": ("OWTFullRun", "owt_valid.txt", "owt_train.txt"),
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def documents(path):
    """Yield bytes ending at a separator, retaining whitespace and the final tail."""
    delimiter = SPECIAL.encode("utf-8")
    pending = b""
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            pieces = (pending + block).split(delimiter)
            pending = pieces.pop()
            for piece in pieces:
                yield piece + delimiter
        if pending:
            yield pending


def encode_dataset(tokenizer, source, output, provenance):
    metadata_path = output.with_suffix(".json")
    raw_path = output.with_suffix(".uint16.partial")
    partial_path = output.with_suffix(".npy.partial")
    for path in (output, metadata_path, raw_path, partial_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")

    started = time.perf_counter()
    last_report = started
    source_stat = source.stat()
    metadata = dict(
        provenance,
        source=str(source),
        source_bytes=source_stat.st_size,
        output=str(output),
        dtype="uint16",
        status="running",
        started_at=datetime.now(UTC).isoformat(),
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    total_bytes = total_tokens = chunks = separators = 0
    source_hash = hashlib.sha256()
    special_id = tokenizer.inverse_vocab[SPECIAL.encode()]
    print(f"START {source.name} -> {output.name}", flush=True)

    try:
        with raw_path.open("xb") as stream:
            for data in documents(source):
                source_hash.update(data)
                text = data.decode("utf-8")
                ids = tokenizer.encode(text)
                if ids and (min(ids) < 0 or max(ids) > 65535):
                    raise ValueError("Token ID outside uint16 range")
                # Verify every chunk, including separator preservation, before saving.
                if tokenizer.decode(ids) != text:
                    raise ValueError(f"Round trip failed at chunk {chunks}")
                expected_separators = text.count(SPECIAL)
                if ids.count(special_id) != expected_separators:
                    raise ValueError(f"Separator count mismatch at chunk {chunks}")
                np.asarray(ids, dtype="<u2").tofile(stream)
                total_bytes += len(data)
                total_tokens += len(ids)
                separators += expected_separators
                chunks += 1
                now = time.perf_counter()
                if now - last_report >= 30:
                    print(
                        f"{source.name}: {total_bytes / source_stat.st_size:.1%}, "
                        f"{total_tokens:,} tokens, {total_bytes / (now - started) / 1e6:.2f} MB/s",
                        flush=True,
                    )
                    last_report = now
        if total_bytes != source_stat.st_size or source.stat().st_mtime_ns != source_stat.st_mtime_ns:
            raise ValueError("Source changed or byte count mismatch")

        # Copy the temporary token stream into a standard, memory-mappable .npy file.
        array = np.lib.format.open_memmap(partial_path, mode="w+", dtype="<u2", shape=(total_tokens,))
        with raw_path.open("rb") as stream:
            offset = 0
            while block := stream.read(8 * 1024 * 1024):
                values = np.frombuffer(block, dtype="<u2")
                array[offset : offset + len(values)] = values
                offset += len(values)
        array.flush()
        del array
        check = np.load(partial_path, mmap_mode="r", allow_pickle=False)
        if check.shape != (total_tokens,) or check.dtype != np.dtype("<u2"):
            raise ValueError("Saved array header mismatch")
        with raw_path.open("rb") as stream:
            offset = 0
            while block := stream.read(8 * 1024 * 1024):
                values = np.frombuffer(block, dtype="<u2")
                if not np.array_equal(check[offset : offset + len(values)], values):
                    raise ValueError("Saved array content mismatch")
                offset += len(values)
        del check
        partial_path.rename(output)
        raw_path.unlink()
        metadata.update(
            status="complete",
            tokens=total_tokens,
            chunks=chunks,
            separators=separators,
            source_sha256=source_hash.hexdigest(),
            elapsed_seconds=time.perf_counter() - started,
            completed_at=datetime.now(UTC).isoformat(),
        )
        print(f"DONE {output.name}: {total_tokens:,} tokens, {total_bytes:,} source bytes", flush=True)
    except BaseException as exc:
        metadata.update(status="failed", error=repr(exc), processed_bytes=total_bytes, tokens=total_tokens)
        raise
    finally:
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, choices=SOURCES)
    parser.add_argument("--split", choices=("valid", "train", "both"), default="both")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/tokenizers/encoded")
    args = parser.parse_args()
    tokenizer_name, valid, train = SOURCES[args.corpus]
    tokenizer_dir = ROOT / "artifacts/tokenizers" / tokenizer_name
    tokenizer = Tokenizer.from_files(
        tokenizer_dir / "vocab.pkl", tokenizer_dir / "merges.pkl", special_tokens=[SPECIAL]
    )
    if min(tokenizer.vocab) < 0 or max(tokenizer.vocab) > 65535:
        raise ValueError("Vocabulary IDs do not fit uint16")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tracked = [
        tokenizer_dir / "vocab.pkl",
        tokenizer_dir / "merges.pkl",
        ROOT / "cs336_basics/tokenizers/tokenizer.py",
        ROOT / "cs336_basics/tokenizers/pretokenization.py",
        Path(__file__).resolve(),
    ]
    provenance = {
        "tokenizer_directory": str(tokenizer_dir),
        "vocab_size": len(tokenizer.vocab),
        "special_tokens": [SPECIAL],
        "sha256": {str(p): digest(p) for p in tracked},
        "chunking": "document boundaries, all source whitespace and separators retained",
    }
    for split, filename in (("valid", valid), ("train", train)):
        if args.split in ("both", split):
            encode_dataset(
                tokenizer, ROOT / "data" / filename, args.output_dir / f"{args.corpus}_{split}.npy", provenance
            )


if __name__ == "__main__":
    main()
