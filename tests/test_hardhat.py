"""Pseudo-label merge for the Hard Hat Workers dataset (no download needed)."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "ph", Path(__file__).resolve().parents[1] / "scripts" / "prepare_hardhat.py")
ph = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ph)


def test_dataset_labels_win_on_heads_model_fills_the_rest():
    gt = [(1, [10, 10, 30, 30]), (0, [0, 0, 50, 200])]                 # helmet, person
    pred = [(7, 0.9, [10, 10, 30, 30]),                                 # "other headwear": dropped
            (0, 0.9, [1, 1, 50, 200]),                                  # same person: dropped
            (0, 0.8, [100, 0, 150, 200]),                               # new person: kept
            (3, 0.7, [5, 50, 45, 120]),                                 # vest: kept
            (8, 0.3, [200, 0, 300, 100])]                               # low confidence: dropped
    out = ph.merge(gt, pred)
    assert sorted(c for c, _ in out) == [0, 0, 1, 3]


def test_yolo_lines_clip_and_drop_tiny():
    lines = ph.yolo_lines([(1, [-10, 0, 20, 20]), (2, [5, 5, 6, 6])], 100, 100)
    assert lines == ["1 0.100000 0.100000 0.200000 0.200000"]


def test_lower_person_threshold_keeps_more_people():
    pred = [(0, 0.35, [100, 0, 150, 200])]
    assert ph.merge([], pred) == []
    assert ph.merge([], pred, person_conf=0.3) == [(0, [100, 0, 150, 200])]
