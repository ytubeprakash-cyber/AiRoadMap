import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

Row = dict[str, Any]


def read_jsonl(path: Path) -> Iterator[Row]:
    """Yield one dict per non-blank line. Must be a generator (use `yield`)."""
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Row]) -> int:
    """Write each row as one JSON line. Return how many rows were written."""
    count = 0
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
            count += 1
    return count


def filter_by_tokens(rows: Iterable[Row], min_tokens: int) -> list[Row]:
    """Keep rows whose "tokens" >= min_tokens."""
    return [row for row in rows if row["tokens"] >= min_tokens]


def answer_lengths(rows: Iterable[Row]) -> dict[str, int]:
    """Map each prompt to len(answer)."""
    return {row["prompt"]: len(row["answer"]) for row in rows}


def batched(items: Iterable[Any], size: int) -> Iterator[list[Any]]:
    """Yield lists of size items, with a smaller final batch if needed."""
    if size < 1:
        raise ValueError("Batch size must be at least 1")

    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []

    if batch:
        yield batch
