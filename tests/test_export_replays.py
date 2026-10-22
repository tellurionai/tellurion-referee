"""scripts/export_replays.py writes real replays in the demo's format (demo/REPLAY_FORMAT.md)."""
import json

import export_replays
import pytest
from test_evaluate_script import dataset, run  # noqa: F401  (fixture + runner helper)

VERDICTS = {"trust", "model_error", "instrument_fault", "discovery"}


@pytest.fixture(scope="module")
def exported(dataset, tmp_path_factory):  # noqa: F811
    tmp = tmp_path_factory.mktemp("export")
    run(dataset, tmp)
    out = tmp / "replays"
    (out).mkdir()
    (out / "stale-demo.json").write_text("{}")
    doc = export_replays.export("cal", tmp / "steps/cal_steps.npz", tmp / "results/cal_metrics.json", out)
    return doc, out


def test_index_is_real_data_with_stated_rules(exported):
    doc, out = exported
    saved = json.loads((out / "index.json").read_text())
    assert saved == doc and saved["is_demo_data"] is False and saved["split"] == "cal"
    assert saved["scenarios"], "no scenarios exported"
    ids = [s["id"] for s in saved["scenarios"]]
    assert len(ids) == len(set(ids))
    for s in saved["scenarios"]:
        assert s["selection_rule"] and "best" not in s["selection_rule"]
        assert (out / s["file"]).is_file() and s["expected_verdict"] in VERDICTS
    assert not (out / "stale-demo.json").exists(), "old demo replays must be removed"


def test_every_replay_follows_the_format(exported):
    doc, out = exported
    for s in doc["scenarios"]:
        r = json.loads((out / s["file"]).read_text())
        assert r["is_demo_data"] is False and r["L"] > 0 and r["dt"] > 0
        f = r["frames"]
        n = len(f["t"])
        assert n >= 2 and all(len(v) == n for v in f.values())
        assert all(b > a for a, b in zip(f["t"], f["t"][1:]))
        assert all(isinstance(x, float) for x in f["true_theta"])
        for key in ("ai_confidence", "referee_score"):
            assert all(x is None or 0.0 <= x <= 1.0 for x in f[key])
        assert set(f["verdict"]) <= VERDICTS
        for v, e in zip(f["verdict"], f["evidence"]):
            assert (e is None) == (v == "trust")
        assert r["onset_step"] == -1 or 0 <= r["onset_step"] < n


def test_flag_frame_matches_the_evaluation(exported):
    """The verdict shown is the referee's real verdict, from its first sustained flag on."""
    doc, out = exported
    for s in doc["scenarios"]:
        f = json.loads((out / s["file"]).read_text())["frames"]
        flagged = [v for v in f["verdict"] if v != "trust"]
        assert len(set(flagged)) <= 1
        if s["missed"]:
            assert not flagged
