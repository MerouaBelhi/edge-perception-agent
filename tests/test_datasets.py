"""Dataset converters (no download needed)."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace as D

spec = importlib.util.spec_from_file_location(
    "pss", Path(__file__).resolve().parents[1] / "scripts" / "prepare_site_safety.py")
pss = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pss)


def test_css_classes_mapped_by_name_whatever_the_order():
    names = ["Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest", "Person",
             "Safety Cone", "Safety Vest", "machinery", "vehicle"]
    m = pss.css_class_map(names)
    out = {names[i]: pss.NAMES[j] for i, j in m.items()}
    assert out == {"Hardhat": "helmet", "Mask": "mask", "NO-Hardhat": "no_helmet",
                   "NO-Mask": "no_mask", "NO-Safety Vest": "no_vest", "Person": "person",
                   "Safety Vest": "vest", "Safety Cone": "safety_cone",
                   "machinery": "machinery", "vehicle": "vehicle"}
    assert pss.css_class_map({0: "person", 1: "hardhat"}) == {0: 0, 1: 1}


def test_open_images_conversion():
    dets = [D(label="Sun hat", bounding_box=[0.4, 0.1, 0.2, 0.1]),
            D(label="Woman", bounding_box=[0.3, 0.1, 0.4, 0.8]),
            D(label="Person", bounding_box=[0.3, 0.1, 0.4, 0.8]),      # same body twice
            D(label="Person", bounding_box=[0.0, 0.0, 1.0, 1.0], IsGroupOf=True),
            D(label="Dog", bounding_box=[0.0, 0.5, 0.2, 0.2])]
    lines = pss.oi_yolo_lines(dets)
    assert [l.split()[0] for l in lines] == ["7", "0"]
    lines = pss.oi_yolo_lines(dets + [D(label="Car", bounding_box=[.5, .5, .3, .2]),
                                      D(label="Vehicle", bounding_box=[.5, .5, .3, .2])])
    assert [l.split()[0] for l in lines] == ["7", "0", "9"]          # one vehicle, not two
    assert lines[0] == "7 0.500000 0.150000 0.200000 0.100000"


def test_open_images_skips_ambiguous_helmets_and_images_without_headwear():
    assert pss.oi_yolo_lines([D(label="Hat", bounding_box=[0, 0, .1, .1]),
                              D(label="Helmet", bounding_box=[.5, 0, .1, .1])]) is None
    assert pss.oi_yolo_lines([D(label="Man", bounding_box=[0, 0, .5, .9])]) is None


def test_css_export_layout(tmp_path):
    root = tmp_path / "css"
    for split in ("train", "valid"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        (root / split / "images" / "a.jpg").write_bytes(b"x")
        (root / split / "labels" / "a.txt").write_text(
            "5 0.5 0.5 0.2 0.6\n6 0.5 0.5 0.1 0.1\n0 0.5 0.2 0.05 0.05\n0 0.1 0.1 0.2 0.1 0.3 0.3\n")
    (root / "data.yaml").write_text("names: ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', "
                                    "'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', "
                                    "'machinery', 'vehicle']\n")
    pss.convert_css(root, tmp_path / "out")
    lbl = (tmp_path / "out" / "labels" / "val" / "css_a.txt").read_text().splitlines()
    assert lbl == ["0 0.5 0.5 0.2 0.6", "10 0.5 0.5 0.1 0.1",       # person, cone
                   "1 0.5 0.2 0.05 0.05"]                          # polygon line dropped


def test_vest_ground_truth_with_and_without_a_no_vest_class():
    spec2 = importlib.util.spec_from_file_location(
        "ev", Path(__file__).resolve().parents[1] / "scripts" / "eval_safety.py")
    ev = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(ev)
    persons = [[0, 0, 100, 300], [200, 0, 300, 300], [400, 0, 500, 300]]
    labels = [("vest", [10, 80, 90, 180]), ("no_vest", [210, 80, 290, 180])]
    assert ev.gt_vest(labels, persons, explicit_no_vest=True) == ["vest", "no_vest", "unlabelled"]
    assert ev.gt_vest(labels, persons, explicit_no_vest=False) == ["vest", "no_vest", "no_vest"]
    _, st = ev.gt_status([("person", [0, 0, 100, 300]), ("other_headwear", [30, 0, 70, 30])])
    assert st == ["other_headwear"]


def test_css_kaggle_layout_without_data_yaml(tmp_path):
    root = tmp_path / "download" / "css-data"
    (root / "train" / "images").mkdir(parents=True)
    (root / "train" / "labels").mkdir(parents=True)
    (root / "train" / "images" / "b.jpg").write_bytes(b"x")
    (root / "train" / "labels" / "b.txt").write_text("7 0.5 0.5 0.3 0.3\n")   # Safety Vest
    pss.convert_css(tmp_path / "download", tmp_path / "out")
    assert (tmp_path / "out" / "labels" / "train" / "css_b.txt").read_text() == "3 0.5 0.5 0.3 0.3"


def test_mask_ground_truth():
    spec2 = importlib.util.spec_from_file_location(
        "ev2", Path(__file__).resolve().parents[1] / "scripts" / "eval_safety.py")
    ev = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(ev)
    persons = [[0, 0, 100, 300], [200, 0, 300, 300]]
    assert ev.gt_mask([("no_mask", [30, 20, 70, 60])], persons) == ["no_mask", "unlabelled"]
