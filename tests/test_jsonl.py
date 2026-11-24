import types
from pathlib import Path

import pytest

from modelcraft_lab.jsonl import (
    answer_lengths,
    batched,
    filter_by_tokens,
    read_jsonl,
    write_jsonl,
)

DATA = Path(__file__).parent.parent / "data" / "sample.jsonl"


def test_read_jsonl_is_a_generator_and_skips_blank_lines():
    rows = read_jsonl(DATA)
    assert isinstance(rows, types.GeneratorType)
    rows = list(rows)
    assert len(rows) == 4
    assert rows[0] == {"prompt": "2+2?", "answer": "4", "tokens": 3}


def test_generator_is_single_use():
    rows = read_jsonl(DATA)
    assert len(list(rows)) == 4
    assert list(rows) == []


def test_write_then_read_roundtrip(tmp_path):
    out = tmp_path / "out.jsonl"
    rows = [{"a": 1}, {"a": 2, "b": "x"}]
    assert write_jsonl(out, rows) == 2
    assert list(read_jsonl(out)) == rows


def test_filter_by_tokens():
    rows = list(read_jsonl(DATA))
    kept = filter_by_tokens(rows, min_tokens=4)
    assert [r["prompt"] for r in kept] == ["Capital of France?", "Color of the sky?"]


def test_answer_lengths():
    rows = read_jsonl(DATA)
    assert answer_lengths(rows) == {
        "2+2?": 1,
        "Capital of France?": 5,
        "Color of the sky?": 4,
        "Largest planet?": 7,
    }


def test_batched():
    result = batched(range(5), 2)
    assert isinstance(result, types.GeneratorType)
    assert list(result) == [[0, 1], [2, 3], [4]]


def test_batched_rejects_bad_size():
    with pytest.raises(ValueError):
        list(batched([1, 2, 3], 0))
