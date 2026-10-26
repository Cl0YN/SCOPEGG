from pathlib import Path

from utility_coach import config


def test_paths_are_inside_repo():
    assert config.ROOT == Path(__file__).resolve().parent.parent
    assert config.DATA == config.ROOT / "data"
    assert config.OUT == config.ROOT / "out"
    assert config.FIG == config.DATA / "figures"


def test_constants_match_contract():
    assert config.TICK_RATE == 64
    assert (config.TEAM_T, config.TEAM_CT) == (2, 3)
    assert config.TEAMFLASH_MIN_S == 1.0
    assert config.MIN_LINEUP_N == 30
    assert config.LINEUP_EPS == 120.0
    assert config.TF_PENALTY_S == 2.0
    assert config.N_ZONES == 24
    assert config.SEED == 42
    assert config.OPENROUTER_MODEL_DEFAULT == "deepseek/deepseek-v4.1-flash"


def test_raw_pickle_env_override(monkeypatch, tmp_path):
    import importlib

    fake = tmp_path / "x.list"
    monkeypatch.setenv("UC_RAW", str(fake))
    reloaded = importlib.reload(config)
    try:
        assert reloaded.RAW_PICKLE == fake
    finally:
        monkeypatch.undo()
        importlib.reload(config)
