import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from utility_coach.features import CATEGORICAL, FEATURES
from utility_coach import model
from utility_coach.model import (baseline_zonepair, evaluate_binary, evaluate_regression, model_categories,
                                 prepare_X, split_matches, train_lgbm)


def make_synthetic_features(n: int = 3000, n_matches: int = 60, seed: int = 0) -> pd.DataFrame:
    """flash_features-подобный фрейм: enemy_blind_s зависит от to_x, teamflash — от to_y."""
    rng = np.random.default_rng(seed)
    match_id = np.repeat(np.arange(1000, 1000 + n_matches), n // n_matches).astype("int64")
    to_x = rng.uniform(-1000, 1000, n)
    to_y = rng.uniform(-1000, 1000, n)
    df = pd.DataFrame({
        "throw_uid": np.arange(n, dtype="int64"),
        "match_id": match_id,
        "thrower_id": (match_id * 10 + rng.integers(0, 10, n)).astype("int64"),
        "throw_x": rng.uniform(-1000, 1000, n).astype("float32"),
        "throw_y": rng.uniform(-1000, 1000, n).astype("float32"),
        "throw_z": rng.uniform(0, 100, n).astype("float32"),
        "yaw": rng.uniform(-180, 180, n).astype("float32"),
        "pitch": rng.uniform(-30, 30, n).astype("float32"),
        "crouch": rng.random(n) < 0.2,
        "airborne": rng.random(n) < 0.1,
        "speed": rng.uniform(0, 250, n).astype("float32"),
        "to_x": to_x.astype("float32"),
        "to_y": to_y.astype("float32"),
        "to_z": rng.uniform(0, 100, n).astype("float32"),
        "dist": rng.uniform(100, 2000, n).astype("float32"),
        "flight_s": rng.uniform(0.5, 2.0, n).astype("float32"),
        "team": pd.Categorical(rng.choice([2, 3], n), categories=[2, 3]),
        "round": rng.integers(1, 25, n).astype("int16"),
        "t_in_round_s": rng.uniform(0, 100, n).astype("float32"),
        "zone_from": pd.Categorical(rng.integers(0, 6, n), categories=list(range(6))),
        "zone_to": pd.Categorical(rng.integers(0, 6, n), categories=list(range(6))),
    })
    df["enemy_blind_s"] = np.clip(1.0 + 3.0 * (to_x > 0) + rng.normal(0, 0.5, n), 0, None).astype("float32")
    df["teamflash"] = rng.random(n) < np.where(to_y > 0, 0.5, 0.05)
    df["self_flash"] = rng.random(n) < 0.2
    df["flash_kill"] = rng.random(n) < 0.1
    df["n_enemies_blinded"] = rng.integers(0, 3, n).astype("int16")
    return df


def test_split_matches_deterministic_disjoint_and_sized():
    ids = list(range(100))
    train, test = split_matches(ids)
    assert len(test) == 20 and len(train) == 80
    assert train.isdisjoint(test) and train | test == set(ids)
    shuffled = list(np.random.default_rng(1).permutation(ids)) + ids[:10]   # порядок и повторы не важны
    assert split_matches(shuffled) == (train, test)
    assert split_matches(ids, seed=7)[1] != test


def test_baseline_zonepair_falls_back_to_global_mean():
    train = pd.DataFrame({"zone_from": [0, 0, 1], "zone_to": [1, 1, 1], "y": [1.0, 3.0, 10.0]})
    test = pd.DataFrame({"zone_from": [0, 1, 2], "zone_to": [1, 1, 2]})
    got = baseline_zonepair(train, test, "y")
    assert got == pytest.approx([2.0, 10.0, 14.0 / 3.0])


def test_baseline_zonepair_accepts_categorical_zones():
    train = pd.DataFrame({"zone_from": pd.Categorical([0, 0]), "zone_to": pd.Categorical([1, 1]),
                          "y": [True, False]})
    test = pd.DataFrame({"zone_from": pd.Categorical([0]), "zone_to": pd.Categorical([1])})
    assert baseline_zonepair(train, test, "y") == pytest.approx([0.5])


def test_evaluate_regression_known_values():
    m = evaluate_regression(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0]))
    assert m["mae"] == pytest.approx(1.0 / 3.0)
    assert m["r2"] == pytest.approx(1.0 - 1.0 / 2.0)


def test_evaluate_binary_known_values():
    y = np.array([0, 0, 1, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8, 1.0])
    m = evaluate_binary(y, p)
    assert m["auc"] == pytest.approx(5.0 / 6.0)          # 5 из 6 пар (pos, neg) упорядочены верно
    assert m["brier"] == pytest.approx((0.01 + 0.16 + 0.4225 + 0.04 + 0.0) / 5)
    assert m["calibration"] == [[0.1, 0.0, 1], [pytest.approx(0.35), 1.0, 1], [0.4, 0.0, 1],
                                [0.8, 1.0, 1], [1.0, 1.0, 1]]   # p=1.0 попадает в последний бин
    assert sum(c[2] for c in m["calibration"]) == len(y)


def test_booster_roundtrip_non_ascii_path(tmp_path):
    # Review Focus: реальный путь репо содержит кириллицу (ДЗ_Мо); модель должна сохраняться и читаться
    df = make_synthetic_features(n=500, n_matches=10)
    X = prepare_X(df, model_categories(df))
    reg = train_lgbm(X, df["enemy_blind_s"].to_numpy(), "regression")
    path = tmp_path / "данные_ДЗ" / "models" / "xflash_blind.txt"
    model.save_booster(reg, path)
    back = model.load_booster(path)
    assert np.allclose(back.predict(X), reg.predict(X))


def test_train_lgbm_learns_signal_with_categories():
    df = make_synthetic_features(n=2000, n_matches=40)
    X = prepare_X(df, model_categories(df))
    for c in CATEGORICAL:
        assert isinstance(X[c].dtype, pd.CategoricalDtype)
    assert list(X.columns) == FEATURES

    reg = train_lgbm(X, df["enemy_blind_s"].to_numpy(), "regression")
    assert isinstance(reg, lgb.Booster)
    pred = reg.predict(X)
    assert np.corrcoef(pred, df["enemy_blind_s"])[0, 1] > 0.8

    clf = train_lgbm(X, df["teamflash"].to_numpy(), "binary")
    p = clf.predict(X)
    assert ((p >= 0) & (p <= 1)).all()
    assert p[df["to_y"].to_numpy() > 0].mean() > p[df["to_y"].to_numpy() <= 0].mean() + 0.2

    with pytest.raises(ValueError):
        train_lgbm(X, df["teamflash"].to_numpy(), "poisson")


from utility_coach.model import run_model  # noqa: E402


@pytest.fixture
def synth_dir(tmp_path: Path) -> Path:
    make_synthetic_features().to_parquet(tmp_path / "flash_features.parquet", index=False)
    return tmp_path


def test_run_model_beats_global_baseline_and_writes_outputs(synth_dir):
    metrics = run_model(synth_dir)

    ho = metrics["holdout"]
    assert ho["xflash_blind"]["lgbm"]["mae"] < ho["xflash_blind"]["baseline_global"]["mae"]
    assert metrics["cv"]["xflash_blind"]["lgbm"]["mae"] < metrics["cv"]["xflash_blind"]["baseline_global"]["mae"]
    assert ho["xflash_tf"]["lgbm"]["auc"] > 0.6
    for part in ("cv", "holdout"):
        for name in ("xflash_blind", "xflash_tf"):
            assert set(metrics[part][name]) == {"target", "lgbm", "baseline_global", "baseline_zonepair"}
    assert set(metrics["importance"]["xflash_blind"]) == set(FEATURES)
    top_blind = max(metrics["importance"]["xflash_blind"], key=metrics["importance"]["xflash_blind"].get)
    assert top_blind == "to_x"

    scored = pd.read_parquet(synth_dir / "flash_scored.parquet")
    assert len(scored) == 3000
    assert scored["xflash_blind"].notna().all() and scored["xflash_tf"].notna().all()
    assert scored["xflash_tf"].between(0, 1).all()
    _, test_ids = split_matches(scored["match_id"].unique())
    assert (scored["is_test"] == scored["match_id"].isin(test_ids)).all()
    assert scored["is_test"].mean() == pytest.approx(0.2, abs=0.01)

    for f in ("models/xflash_blind.txt", "models/xflash_tf.txt", "figures/calibration.png",
              "figures/importance.png", "model_metrics.json"):
        assert (synth_dir / f).exists(), f
    on_disk = json.loads((synth_dir / "model_metrics.json").read_text(encoding="utf-8"))
    assert on_disk["holdout"]["xflash_tf"]["lgbm"]["calibration"]
    booster = model.load_booster(synth_dir / "models" / "xflash_blind.txt")
    assert booster.num_trees() == model.N_ROUNDS


class SpyBooster:
    """Запоминает строки обучения и строки, для которых делал предсказания."""

    def __init__(self, train_rows: np.ndarray):
        self.train_rows = np.asarray(train_rows)
        self.predicted_rows: list[np.ndarray] = []

    def predict(self, X, pred_contrib: bool = False):
        self.predicted_rows.append(X.index.to_numpy())
        if pred_contrib:
            return np.zeros((len(X), X.shape[1] + 1))
        return np.full(len(X), 0.5)

    def save_model(self, path: str):
        Path(path).write_text("spy", encoding="utf-8")

    def model_to_string(self) -> str:
        return "spy"


def test_run_model_oof_never_predicts_rows_of_matches_seen_in_training(synth_dir, monkeypatch):
    boosters: list[SpyBooster] = []

    def spy_train(X, y, objective, seed=None):
        b = SpyBooster(X.index.to_numpy())
        boosters.append(b)
        return b

    monkeypatch.setattr(model, "train_lgbm", spy_train)
    run_model(synth_dir)

    df = pd.read_parquet(synth_dir / "flash_scored.parquet")
    match = df["match_id"].to_numpy()
    is_test = df["is_test"].to_numpy()
    assert len(boosters) == 2 * (model.N_FOLDS + 1)          # 5 фолдов + финальная, на 2 таргета

    covered = []
    for b in boosters:
        trained = set(match[b.train_rows])
        assert not is_test[b.train_rows].any()                 # test-матчи никогда не идут в обучение
        for rows in b.predicted_rows:
            assert trained.isdisjoint(set(match[rows]))        # предсказываем только невиданные матчи
            covered.append(rows)
    covered = np.unique(np.concatenate(covered))
    assert len(covered) == len(df)                             # каждая флешка получила предсказание


# --- final fix F5 --------------------------------------------------------------------------------
class NegativeBooster(SpyBooster):
    """Регрессия, которая предсказывает отрицательное ослепление."""

    def predict(self, X, pred_contrib: bool = False):
        if pred_contrib:
            return np.zeros((len(X), X.shape[1] + 1))
        return np.full(len(X), -0.7)


def test_predict_clips_negative_blind():
    X = pd.DataFrame({"a": [1.0, 2.0]})
    assert model.predict(NegativeBooster([]), X, "regression").tolist() == [0.0, 0.0]
    assert model.predict(NegativeBooster([]), X, "binary").tolist() == [-0.7, -0.7]  # tf не трогаем


def test_run_model_never_persists_negative_xflash_blind(synth_dir, monkeypatch):
    monkeypatch.setattr(model, "train_lgbm", lambda X, y, objective, seed=None: NegativeBooster(X.index))
    metrics = run_model(synth_dir)
    scored = pd.read_parquet(synth_dir / "flash_scored.parquet")
    assert (scored["xflash_blind"] >= 0).all()
    assert (scored["xflash_blind"] == 0).all()
    y = scored["enemy_blind_s"].to_numpy(dtype="float64")
    te = scored["is_test"].to_numpy()
    # метрики считаются по обрезанному прогнозу: MAE = среднее |y - 0|
    assert metrics["holdout"]["xflash_blind"]["lgbm"]["mae"] == pytest.approx(np.abs(y[te]).mean(), rel=1e-6)
    assert metrics["cv"]["xflash_blind"]["lgbm"]["mae"] == pytest.approx(np.abs(y[~te]).mean(), rel=1e-6)
