"""Таргеты и признаки на одну флешку (только grenade_type == "flash")."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from utility_coach.config import DATA, TEAM_CT, TEAM_T, TEAMFLASH_MIN_S, TICK_RATE
from utility_coach.zones import assign_zones, load_centers

FEATURES: list[str] = ["throw_x", "throw_y", "throw_z", "yaw", "pitch", "crouch", "airborne", "speed",
                       "to_x", "to_y", "to_z", "dist", "flight_s", "team", "round", "t_in_round_s",
                       "zone_from", "zone_to"]
CATEGORICAL: list[str] = ["team", "zone_from", "zone_to"]
TARGETS: list[str] = ["enemy_blind_s", "teamflash", "self_flash", "flash_kill", "n_enemies_blinded"]
ID_COLS: list[str] = ["throw_uid", "match_id", "thrower_id"]

# колонки throws.parquet, которые реально нужны признакам (экономия RAM при чтении)
THROW_COLS: list[str] = ["throw_uid", "match_id", "round", "thrower_id", "team", "start_tick", "end_tick",
                         "throw_x", "throw_y", "throw_z", "yaw", "pitch", "vel_x", "vel_y",
                         "to_x", "to_y", "to_z", "crouch", "airborne"]
VICTIM_COLS: list[str] = ["throw_uid", "victim_id", "victim_team", "flash_duration", "killed"]


def round_start_ticks(throws: pd.DataFrame) -> pd.DataFrame:
    """[match_id, round, round_start_tick]: минимальный start_tick среди всех гранат раунда."""
    rs = throws.groupby(["match_id", "round"], as_index=False, sort=True)["start_tick"].min()
    return rs.rename(columns={"start_tick": "round_start_tick"})


def flash_targets(flash_throws: pd.DataFrame, victims: pd.DataFrame) -> pd.DataFrame:
    """Таргеты и описательные метрики на флешку, index = throw_uid (в порядке flash_throws)."""
    index = pd.Index(flash_throws["throw_uid"].to_numpy(dtype="int64"), name="throw_uid")
    thrower = flash_throws[["throw_uid", "thrower_id", "team"]]
    v = victims[VICTIM_COLS].merge(thrower, on="throw_uid", how="inner")

    dur = v["flash_duration"].to_numpy(dtype="float64")
    enemy = v["victim_team"].to_numpy(dtype="int64") != v["team"].to_numpy(dtype="int64")
    is_self = v["victim_id"].to_numpy(dtype="int64") == v["thrower_id"].to_numpy(dtype="int64")
    mate = ~enemy & ~is_self

    per = pd.DataFrame({
        "throw_uid": v["throw_uid"].to_numpy(dtype="int64"),
        "enemy_blind_s": np.where(enemy, dur, 0.0),
        "teamflash": mate & (dur > TEAMFLASH_MIN_S),
        "self_flash": is_self & (dur > 0.0),
        "flash_kill": enemy & v["killed"].to_numpy(dtype=bool),
        "n_enemies_blinded": (enemy & (dur > TEAMFLASH_MIN_S)).astype("int64"),
    })
    agg = per.groupby("throw_uid").agg(
        enemy_blind_s=("enemy_blind_s", "sum"),
        teamflash=("teamflash", "any"),
        self_flash=("self_flash", "any"),
        flash_kill=("flash_kill", "any"),
        n_enemies_blinded=("n_enemies_blinded", "sum"),
    ).reindex(index)

    out = pd.DataFrame(index=index)
    out["enemy_blind_s"] = agg["enemy_blind_s"].fillna(0.0).astype("float32")
    for col in ("teamflash", "self_flash", "flash_kill"):
        out[col] = agg[col].eq(True).to_numpy()   # NaN (нет жертв) -> False без FutureWarning
    out["n_enemies_blinded"] = agg["n_enemies_blinded"].fillna(0).astype("int16")
    return out


def flash_features(flash_throws: pd.DataFrame, round_starts: pd.DataFrame, centers: np.ndarray,
                   z_scale: float = 3.0) -> pd.DataFrame:
    """ID_COLS + FEATURES в порядке строк flash_throws. team/zone_from/zone_to — pandas category."""
    ft = flash_throws.reset_index(drop=True)
    start = ft["start_tick"].to_numpy(dtype="int64")
    end = ft["end_tick"].to_numpy(dtype="int64")
    rst = (ft[["match_id", "round"]]
           .merge(round_starts, on=["match_id", "round"], how="left")["round_start_tick"]
           .to_numpy(dtype="float64"))
    rst = np.where(np.isnan(rst), start, rst)          # раунд без старта -> t_in_round_s = 0

    out = pd.DataFrame({c: ft[c].to_numpy(dtype="int64") for c in ID_COLS})
    for c in ("throw_x", "throw_y", "throw_z", "yaw", "pitch"):
        out[c] = ft[c].to_numpy(dtype="float32")
    out["crouch"] = ft["crouch"].to_numpy(dtype=bool)
    out["airborne"] = ft["airborne"].to_numpy(dtype=bool)
    out["speed"] = np.hypot(ft["vel_x"].to_numpy(dtype="float64"),
                            ft["vel_y"].to_numpy(dtype="float64")).astype("float32")
    for c in ("to_x", "to_y", "to_z"):
        out[c] = ft[c].to_numpy(dtype="float32")

    throw_xyz = ft[["throw_x", "throw_y", "throw_z"]].to_numpy(dtype="float64")
    to_xyz = ft[["to_x", "to_y", "to_z"]].to_numpy(dtype="float64")
    out["dist"] = np.sqrt(((to_xyz - throw_xyz) ** 2).sum(axis=1)).astype("float32")
    out["flight_s"] = ((end - start) / TICK_RATE).astype("float32")
    out["team"] = pd.Categorical(ft["team"].to_numpy(dtype="int64"), categories=[TEAM_T, TEAM_CT])
    out["round"] = ft["round"].to_numpy(dtype="int16")
    out["t_in_round_s"] = np.clip((start - rst) / TICK_RATE, 0.0, None).astype("float32")

    zone_cats = list(range(len(centers)))
    out["zone_from"] = pd.Categorical(np.asarray(assign_zones(centers, throw_xyz, z_scale=z_scale), dtype="int64"),
                                      categories=zone_cats)
    out["zone_to"] = pd.Categorical(np.asarray(assign_zones(centers, to_xyz, z_scale=z_scale), dtype="int64"),
                                    categories=zone_cats)
    return out[ID_COLS + FEATURES]


def run_features(data: Path = DATA) -> None:
    """throws.parquet + flash_victims.parquet + zones.json -> flash_features.parquet."""
    data = Path(data)
    throws_path = data / "throws.parquet"
    round_starts = round_start_ticks(pd.read_parquet(throws_path, columns=["match_id", "round", "start_tick"]))
    flash = pd.read_parquet(throws_path, columns=THROW_COLS, filters=[("grenade_type", "==", "flash")])
    victims = pd.read_parquet(data / "flash_victims.parquet", columns=VICTIM_COLS)

    zones_path = data / "zones.json"
    centers = load_centers(zones_path)
    z_scale = float(json.loads(zones_path.read_text(encoding="utf-8")).get("z_scale", 3.0))

    feats = flash_features(flash, round_starts, centers, z_scale=z_scale)
    targets = flash_targets(flash, victims)
    del flash, victims
    out = feats.join(targets, on="throw_uid")
    out.to_parquet(data / "flash_features.parquet", index=False)
    print(f"flash_features: {len(out)} флешек, enemy_blind_s mean={out['enemy_blind_s'].mean():.3f} с, "
          f"teamflash={out['teamflash'].mean():.3f}, self_flash={out['self_flash'].mean():.3f}, "
          f"flash_kill={out['flash_kill'].mean():.3f}, доля без эффекта на врагов="
          f"{(out['enemy_blind_s'] == 0).mean():.3f}")
