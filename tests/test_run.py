import pytest

from utility_coach import run

EXPECTED = ["ingest", "zones", "features", "model", "lineups", "rating", "findings", "advise", "report"]


def _record(monkeypatch):
    calls = []

    def fake_resolve(step):
        def fn(**kwargs):
            calls.append((step, kwargs))
        return fn

    monkeypatch.setattr(run, "_resolve", fake_resolve)
    return calls


def test_all_runs_steps_in_order(monkeypatch):
    calls = _record(monkeypatch)
    assert run.main(["all", "--limit", "3"]) == 0
    assert run.STEPS == EXPECTED
    assert [s for s, _ in calls] == EXPECTED
    assert calls[0] == ("ingest", {"limit": 3})
    assert all(kwargs == {} for _, kwargs in calls[1:])


def test_single_step(monkeypatch):
    calls = _record(monkeypatch)
    assert run.main(["report"]) == 0
    assert calls == [("report", {})]


def test_ingest_without_limit(monkeypatch):
    calls = _record(monkeypatch)
    run.main(["ingest"])
    assert calls == [("ingest", {"limit": None})]


def test_unknown_step_rejected():
    with pytest.raises(SystemExit):
        run.main(["bogus"])


def test_runners_resolve_to_real_functions():
    for step in EXPECTED:
        assert callable(run._resolve(step))
