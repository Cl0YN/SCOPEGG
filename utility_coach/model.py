"""xFlash: LightGBM против бейзлайнов, GroupKFold по матчам на train, hold-out по матчам."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import brier_score_loss, mean_absolute_error, r2_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from utility_coach.config import DATA, SEED  # noqa: E402
from utility_coach.features import CATEGORICAL, FEATURES  # noqa: E402

N_ROUNDS = 400
N_FOLDS = 5
MAX_CONTRIB_ROWS = 50_000
LGB_PARAMS: dict = {
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,      # = min_child_samples
    "feature_fraction": 0.9,
    "deterministic": True,
    "force_row_wise": True,
    "verbosity": -1,
}
# имя модели -> (таргет, objective); имена совпадают с колонками flash_scored и файлами models/*.txt
MODELS: dict[str, tuple[str, str]] = {
    "xflash_blind": ("enemy_blind_s", "regression"),
    "xflash_tf": ("teamflash", "binary"),
}


def to_jsonable(obj):
    """numpy-скаляры -> python, NaN/inf -> None (json.dump иначе пишет невалидный NaN)."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    return obj


def split_matches(match_ids: Iterable[int], test_frac: float = 0.2, seed: int = SEED) -> tuple[set[int], set[int]]:
    """(train, test) по матчам; результат не зависит от порядка и повторов во входе."""
    ids = np.array(sorted({int(m) for m in match_ids}), dtype=np.int64)
    perm = np.random.default_rng(seed).permutation(ids)
    n_test = int(round(len(ids) * test_frac))
    test = {int(m) for m in perm[:n_test]}
    train = {int(m) for m in perm[n_test:]}
    return train, test


def model_categories(df: pd.DataFrame) -> dict[str, list[int]]:
    """Фиксированный набор категорий по всему датасету, чтобы train/val/test кодировались одинаково."""
    return {c: sorted(int(v) for v in pd.unique(df[c].astype("int64"))) for c in CATEGORICAL}


def prepare_X(df: pd.DataFrame, categories: dict[str, list[int]]) -> pd.DataFrame:
    X = df[FEATURES].copy()
    for c in ("crouch", "airborne"):
        X[c] = X[c].astype("int8")
    for c in CATEGORICAL:
        X[c] = pd.Categorical(df[c].astype("int64").to_numpy(), categories=categories[c])
    return X


def train_lgbm(X: pd.DataFrame, y: np.ndarray, objective: str, seed: int = SEED) -> lgb.Booster:
    """objective: "regression" | "binary". Категориальные признаки берутся из pandas category."""
    if objective not in ("regression", "binary"):
        raise ValueError(f"unknown objective {objective!r}")
    params = dict(LGB_PARAMS, objective=objective, seed=seed)
    ds = lgb.Dataset(X, label=np.asarray(y, dtype="float64"), free_raw_data=True)
    return lgb.train(params, ds, num_boost_round=N_ROUNDS)


def predict(booster, X: pd.DataFrame, objective: str) -> np.ndarray:
    """Единственная точка получения прогнозов. Регрессия ослепления обрезается снизу нулём:
    отрицательных секунд не бывает. Вероятности (binary) LightGBM и так даёт в [0, 1]."""
    pred = np.asarray(booster.predict(X), dtype="float64")
    return np.clip(pred, 0.0, None) if objective == "regression" else pred


def save_booster(booster: lgb.Booster, path: Path) -> None:
    """Пишет модель через Python, чтобы не зависеть от обработки путей в C++-части LightGBM (путь репо содержит кириллицу).
    На lightgbm 4.6.0 save_model с кириллицей тоже работает (проверено), это страховка от других версий."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(booster.model_to_string(), encoding="utf-8")


def load_booster(path: Path) -> lgb.Booster:
    return lgb.Booster(model_str=path.read_text(encoding="utf-8"))


def baseline_zonepair(train: pd.DataFrame, test: pd.DataFrame, target: str) -> np.ndarray:
    """Среднее таргета по паре (zone_from, zone_to) на train; невиданная пара -> глобальное среднее train."""
    tr = pd.DataFrame({"zf": train["zone_from"].astype("int64").to_numpy(),
                       "zt": train["zone_to"].astype("int64").to_numpy(),
                       "y": train[target].astype("float64").to_numpy()})
    global_mean = float(tr["y"].mean())
    means = tr.groupby(["zf", "zt"], as_index=False)["y"].mean()
    te = pd.DataFrame({"zf": test["zone_from"].astype("int64").to_numpy(),
                       "zt": test["zone_to"].astype("int64").to_numpy()})
    return te.merge(means, on=["zf", "zt"], how="left")["y"].fillna(global_mean).to_numpy(dtype="float64")


def evaluate_regression(y: np.ndarray, pred: np.ndarray) -> dict:
    y = np.asarray(y, dtype="float64")
    pred = np.asarray(pred, dtype="float64")
    return {"mae": float(mean_absolute_error(y, pred)), "r2": float(r2_score(y, pred))}


def evaluate_binary(y: np.ndarray, p: np.ndarray) -> dict:
    """AUC, Brier и калибровка по 10 равным бинам [0,0.1),...,[0.9,1.0]; пустые бины не выводятся."""
    y = np.asarray(y, dtype="float64")
    p = np.clip(np.asarray(p, dtype="float64"), 0.0, 1.0)
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    bins = np.minimum((p * 10).astype(int), 9)
    calibration = []
    for b in range(10):
        m = bins == b
        if m.any():
            calibration.append([float(p[m].mean()), float(y[m].mean()), int(m.sum())])
    return {"auc": auc, "brier": float(brier_score_loss(y, p)), "calibration": calibration}


def _plot_calibration(holdout_tf: dict, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1, label="идеал")
    for name, style in (("lgbm", "o-"), ("baseline_zonepair", "s--")):
        cal = holdout_tf[name]["calibration"]
        if cal:
            ax.plot([c[0] for c in cal], [c[1] for c in cal], style, label=name)
    ax.set_xlabel("предсказанная вероятность тимфлеша")
    ax.set_ylabel("фактическая доля тимфлешей")
    ax.set_title("Калибровка xflash_tf (hold-out)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def _plot_importance(importance: dict[str, dict[str, float]], out_png: Path) -> None:
    fig, axes = plt.subplots(1, len(importance), figsize=(6 * len(importance), 6))
    axes = np.atleast_1d(axes)
    titles = {"xflash_blind": "xflash_blind, |вклад| в секундах",
              "xflash_tf": "xflash_tf, |вклад| в лог-шансах"}
    for ax, (name, imp) in zip(axes, importance.items()):
        items = sorted(imp.items(), key=lambda kv: kv[1])
        ax.barh([k for k, _ in items], [v for _, v in items], color="#4C72B0")
        ax.set_title(titles.get(name, name))
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def run_model(data: Path = DATA) -> dict:
    data = Path(data)
    df = pd.read_parquet(data / "flash_features.parquet")
    categories = model_categories(df)
    X = prepare_X(df, categories)
    base = df[["zone_from", "zone_to"] + [t for t, _ in MODELS.values()]]

    match = df["match_id"].to_numpy(dtype="int64")
    train_ids, test_ids = split_matches(np.unique(match))
    is_test = np.isin(match, np.array(sorted(test_ids), dtype=np.int64))
    tr = np.flatnonzero(~is_test)
    te = np.flatnonzero(is_test)
    folds = list(GroupKFold(n_splits=N_FOLDS).split(tr, groups=match[tr]))

    models_dir = data / "models"
    fig_dir = data / "figures"
    models_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    metrics: dict = {
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "n_train_matches": len(train_ids), "n_test_matches": len(test_ids),
        "params": dict(LGB_PARAMS, num_boost_round=N_ROUNDS, n_folds=N_FOLDS, seed=SEED),
        "cv": {}, "holdout": {}, "importance": {},
    }
    preds: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(SEED)
    contrib_rows = te if len(te) <= MAX_CONTRIB_ROWS else np.sort(rng.choice(te, MAX_CONTRIB_ROWS, replace=False))

    for name, (target, objective) in MODELS.items():
        y = df[target].to_numpy(dtype="float64")
        pred = np.full(len(df), np.nan)
        zp_cv = np.full(len(tr), np.nan)
        gl_cv = np.full(len(tr), np.nan)
        for fold, (a, b) in enumerate(folds):
            fit, val = tr[a], tr[b]
            booster = train_lgbm(X.iloc[fit], y[fit], objective)
            pred[val] = predict(booster, X.iloc[val], objective)
            zp_cv[b] = baseline_zonepair(base.iloc[fit], base.iloc[val], target)
            gl_cv[b] = y[fit].mean()
            print(f"{name}: fold {fold + 1}/{N_FOLDS} готов")

        final = train_lgbm(X.iloc[tr], y[tr], objective)
        save_booster(final, models_dir / f"{name}.txt")
        pred[te] = predict(final, X.iloc[te], objective)
        zp_te = baseline_zonepair(base.iloc[tr], base.iloc[te], target)
        gl_te = np.full(len(te), y[tr].mean())

        ev = evaluate_regression if objective == "regression" else evaluate_binary
        metrics["cv"][name] = {"target": target, "lgbm": ev(y[tr], pred[tr]),
                               "baseline_global": ev(y[tr], gl_cv), "baseline_zonepair": ev(y[tr], zp_cv)}
        metrics["holdout"][name] = {"target": target, "lgbm": ev(y[te], pred[te]),
                                    "baseline_global": ev(y[te], gl_te), "baseline_zonepair": ev(y[te], zp_te)}

        contrib = np.asarray(final.predict(X.iloc[contrib_rows], pred_contrib=True))
        imp = np.abs(contrib[:, :len(FEATURES)]).mean(axis=0)   # последний столбец — bias
        metrics["importance"][name] = {f: float(v) for f, v in sorted(zip(FEATURES, imp), key=lambda kv: -kv[1])}
        preds[name] = pred
        print(f"{name}: holdout lgbm={metrics['holdout'][name]['lgbm']}")

    scored = df
    for name in MODELS:
        scored[name] = preds[name].astype("float32")
    scored["is_test"] = is_test
    scored.to_parquet(data / "flash_scored.parquet", index=False)

    (data / "model_metrics.json").write_text(
        json.dumps(to_jsonable(metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    _plot_calibration(metrics["holdout"]["xflash_tf"], fig_dir / "calibration.png")
    _plot_importance(metrics["importance"], fig_dir / "importance.png")
    return metrics
