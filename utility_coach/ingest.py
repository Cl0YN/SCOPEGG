"""pickle -> parquet: throws, flash_victims, damage_events, players."""
from __future__ import annotations

import math
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from utility_coach.config import DATA, RAW_PICKLE
from utility_coach.safe_pickle import Stop, stream_matches

# Порядок обхода гранат внутри матча: от него зависит сквозной throw_uid.
GRENADE_KEYS: list[tuple[str, str]] = [
    ("smoke", "Smokes"), ("flash", "Flashbangs"), ("he", "HEs"), ("fire", "Fires"),
]
FLUSH_EVERY = 100  # матчей в одной порции записи

_F32 = pa.float32()
SCHEMAS: dict[str, pa.Schema] = {
    "throws": pa.schema([
        ("throw_uid", pa.int64()), ("match_id", pa.int64()), ("grenade_type", pa.string()),
        ("projectile_id", pa.int64()), ("round", pa.int16()), ("thrower_id", pa.int64()),
        ("team", pa.int8()), ("start_tick", pa.int32()), ("end_tick", pa.int32()),
        ("throw_x", _F32), ("throw_y", _F32), ("throw_z", _F32), ("yaw", _F32), ("pitch", _F32),
        ("vel_x", _F32), ("vel_y", _F32), ("vel_z", _F32),
        ("to_x", _F32), ("to_y", _F32), ("to_z", _F32),
        ("crouch", pa.bool_()), ("airborne", pa.bool_()), ("inferno_ok", pa.bool_()),
    ]),
    "flash_victims": pa.schema([
        ("throw_uid", pa.int64()), ("match_id", pa.int64()), ("victim_id", pa.int64()),
        ("victim_team", pa.int8()), ("flash_duration", _F32), ("killed", pa.bool_()),
        ("killed_by", pa.int64()), ("killed_tick", pa.int32()),
    ]),
    "damage_events": pa.schema([
        ("throw_uid", pa.int64()), ("match_id", pa.int64()), ("grenade_type", pa.string()),
        ("victim_id", pa.int64()), ("victim_team", pa.int8()), ("health_dmg", pa.int32()),
        ("armor_dmg", pa.int32()), ("killed", pa.bool_()),
    ]),
    "players": pa.schema([
        ("match_id", pa.int64()), ("player_id", pa.int64()), ("rating", _F32),
    ]),
}


def _throw_row(uid: int, match_id: int, gtype: str, g: dict) -> dict:
    gi = g["GeneralInfo"]
    tp = gi["ThrowPoint"]
    pos, vel, to = tp["Position"], tp["Velocity"], gi["PosTo"]
    if gtype == "fire":
        inferno_ok = int((g.get("FinalInfernoInfo") or {}).get("UniqueID", 0)) != 0
    else:
        inferno_ok = True
    return {
        "throw_uid": uid, "match_id": match_id, "grenade_type": gtype,
        "projectile_id": int(gi["ProjectileID"]), "round": int(gi["RoundIndexMatch"]),
        "thrower_id": int(gi["ThrowerID"]), "team": int(gi["Team"]),
        "start_tick": int(gi["StartTick"]), "end_tick": int(gi["EndTick"]),
        "throw_x": pos["X"], "throw_y": pos["Y"], "throw_z": pos["Z"],
        "yaw": tp["ViewX"], "pitch": tp["ViewY"],
        "vel_x": vel["X"], "vel_y": vel["Y"], "vel_z": vel["Z"],
        "to_x": to["X"], "to_y": to["Y"], "to_z": to["Z"],
        "crouch": bool(tp["IsCrouching"]), "airborne": bool(tp["IsAirborne"]),
        "inferno_ok": inferno_ok,
    }


def _victim_rows(uid: int, match_id: int, g: dict) -> list[dict]:
    rows = []
    for v in g.get("FlashedPlayers") or []:
        wk = v.get("WasKilled")
        rows.append({
            "throw_uid": uid, "match_id": match_id, "victim_id": int(v["PlayerID"]),
            "victim_team": int(v["Team"]), "flash_duration": float(v["FlashDuration"]),
            "killed": bool(wk),
            "killed_by": int(wk["By"]) if wk else -1,
            "killed_tick": int(wk["TickNumber"]) if wk else -1,
        })
    return rows


def _damage_rows(uid: int, match_id: int, gtype: str, g: dict) -> list[dict]:
    per_victim: dict[int, dict] = {}
    for d in g.get("DamagedPlayers") or []:
        vid = int(d["PlayerID"])
        row = per_victim.setdefault(vid, {
            "throw_uid": uid, "match_id": match_id, "grenade_type": gtype, "victim_id": vid,
            "victim_team": int(d["Team"]), "health_dmg": 0, "armor_dmg": 0, "killed": False,
        })
        for e in d.get("Damage") or []:
            row["health_dmg"] += int(e.get("HealthDamage", 0))
            row["armor_dmg"] += int(e.get("ArmorDamage", 0))
            row["killed"] = row["killed"] or bool(e.get("WasKilled"))
    return list(per_victim.values())


def flatten_match(m: dict, uid_start: int) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Раскладывает один матч в строки 4 таблиц; throw_uid = uid_start + i."""
    match_id = int(m["MatchID"])
    throws, victims, damage = [], [], []
    uid = uid_start
    for gtype, key in GRENADE_KEYS:
        for g in m.get(key) or []:
            throws.append(_throw_row(uid, match_id, gtype, g))
            if gtype == "flash":
                victims.extend(_victim_rows(uid, match_id, g))
            elif gtype in ("he", "fire"):
                damage.extend(_damage_rows(uid, match_id, gtype, g))
            uid += 1
    players = []
    for p in m.get("Players") or []:
        r = p.get("PremierRatingNew")
        players.append({"match_id": match_id, "player_id": int(p["StaticInfo"]["PlayerID"]),
                        "rating": float(r) if r is not None else math.nan})
    return throws, victims, damage, players


def run_ingest(raw: Path = RAW_PICKLE, data: Path = DATA, limit: int | None = None) -> dict[str, int]:
    """Пишет 4 parquet порциями через ParquetWriter; возвращает число строк по таблицам и матчей."""
    data = Path(data)
    data.mkdir(parents=True, exist_ok=True)
    names = list(SCHEMAS)
    tmp = {n: data / f"{n}.parquet.tmp" for n in names}
    writers = {n: pq.ParquetWriter(tmp[n], SCHEMAS[n], compression="zstd") for n in names}
    buf: dict[str, list[dict]] = {n: [] for n in names}
    counts = {n: 0 for n in names}
    state = {"uid": 0, "matches": 0}
    t0 = time.time()

    def flush():
        for n in names:
            if buf[n]:
                writers[n].write_table(pa.Table.from_pylist(buf[n], schema=SCHEMAS[n]))
                counts[n] += len(buf[n])
                buf[n] = []

    def on_match(m: dict) -> None:
        if limit is not None and state["matches"] >= limit:
            raise Stop
        parts = flatten_match(m, state["uid"])
        for n, rows in zip(names, parts):
            buf[n].extend(rows)
        state["uid"] += len(parts[0])
        state["matches"] += 1
        if state["matches"] % FLUSH_EVERY == 0:
            flush()
        if state["matches"] % 1000 == 0:
            print(f"ingest: {state['matches']} matches, {state['uid']} throws, {time.time() - t0:.0f}s",
                  flush=True)
        if limit is not None and state["matches"] >= limit:
            raise Stop

    ok = False
    try:
        stream_matches(Path(raw), on_match)
        flush()
        ok = True
    finally:
        for w in writers.values():
            w.close()
        if not ok:
            for p in tmp.values():
                p.unlink(missing_ok=True)
    for n in names:
        tmp[n].replace(data / f"{n}.parquet")
    return {"matches": state["matches"], **counts}
