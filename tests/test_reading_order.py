from __future__ import annotations

from app.engine.reading_order import (
    assign_columns,
    detect_column_gaps,
    reorder_blocks,
)


def _make_block(bbox, text="block", type="text"):
    return {
        "type": type,
        "text": text,
        "page": 0,
        "bbox": bbox,
        "font_size": 12,
        "font": "Test",
    }


class TestDetectColumnGaps:
    def test_single_column_no_gaps(self):
        blocks = [
            _make_block((0, 0, 100, 50)),
            _make_block((0, 60, 100, 100)),
            _make_block((0, 110, 100, 150)),
        ]
        assert detect_column_gaps(blocks) == []

    def test_two_columns_with_gap(self):
        blocks = [
            _make_block((0, 0, 200, 50)),
            _make_block((300, 0, 500, 50)),
            _make_block((0, 60, 200, 100)),
            _make_block((300, 60, 500, 100)),
        ]
        gaps = detect_column_gaps(blocks)
        assert len(gaps) == 1
        assert 200 < gaps[0] < 300

    def test_few_blocks_returns_empty(self):
        assert detect_column_gaps([_make_block((0, 0, 100, 50))]) == []


class TestAssignColumns:
    def test_two_columns(self):
        blocks = [
            _make_block((0, 0, 100, 50)),
            _make_block((300, 0, 400, 50)),
        ]
        cols = assign_columns(blocks, [250])
        assert len(cols) == 2
        assert len(cols[0]) == 1
        assert len(cols[1]) == 1

    def test_three_columns(self):
        blocks = [
            _make_block((0, 0, 100, 50)),
            _make_block((300, 0, 400, 50)),
            _make_block((600, 0, 700, 50)),
        ]
        cols = assign_columns(blocks, [200, 500])
        assert len(cols) == 3
        assert all(len(cols[i]) == 1 for i in range(3))

    def test_non_text_blocks_filtered(self):
        blocks = [
            _make_block((0, 0, 100, 50)),
            {"type": "image", "page": 0, "bbox": (300, 0, 400, 50)},
        ]
        cols = assign_columns(blocks, [250])
        assert len(cols) == 1
        assert len(cols[0]) == 1


class TestReorderBlocks:
    def test_single_column_unchanged(self):
        blocks = [
            _make_block((0, 0, 100, 20), text="A"),
            _make_block((0, 30, 100, 50), text="B"),
        ]
        result = reorder_blocks(blocks)
        assert [b["text"] for b in result] == ["A", "B"]

    def test_two_column_reorder(self):
        blocks = [
            _make_block((300, 30, 400, 50), text="col2_first"),
            _make_block((0, 0, 100, 20), text="col1_first"),
            _make_block((0, 30, 100, 50), text="col1_second"),
        ]
        result = reorder_blocks(blocks, [200])
        texts = [b["text"] for b in result]
        assert texts.index("col1_first") < texts.index("col2_first")
        assert texts.index("col1_second") < texts.index("col2_first")

    def test_preserves_non_text_blocks(self):
        blocks = [
            _make_block((0, 0, 100, 50), text="A"),
            {"type": "image", "page": 0, "bbox": (300, 0, 400, 50), "width": 100, "height": 50},
        ]
        result = reorder_blocks(blocks, [200])
        assert len(result) == 2
