"""Фикстурный мини-pickle в формате реального grenades_by_match_0726.list.

Фейковые классы bson регистрируются в sys.modules, поэтому pickle пишет глобалы
bson.int64.Int64 и bson.objectid.ObjectId ровно так же, как в реальном файле.
Каждый вызов make_* создаёт новые объекты: общие подобъекты между матчами pickle
записал бы ссылкой на memo, а safe_pickle чистит memo между пачками матчей.
"""
from __future__ import annotations

import copy
import copyreg
import pickle
import sys
import types
from pathlib import Path


class FakeInt64(int):
    def __reduce_ex__(self, protocol):
        # Как в реальном файле: NEWOBJ(Int64, (value,)) и BUILD с пустым dict.
        return (copyreg.__newobj__, (type(self), int(self)), {})


FakeInt64.__module__ = "bson.int64"
FakeInt64.__name__ = "Int64"
FakeInt64.__qualname__ = "Int64"


class FakeObjectId:
    __slots__ = ("_oid",)

    def __init__(self, oid: bytes):
        self._oid = oid

    def __getstate__(self):
        return self._oid

    def __setstate__(self, state):
        self._oid = state


FakeObjectId.__module__ = "bson.objectid"
FakeObjectId.__name__ = "ObjectId"
FakeObjectId.__qualname__ = "ObjectId"


def _register_fake_bson() -> None:
    bson = sys.modules.get("bson")
    if bson is None or not getattr(bson, "_uc_fake", False):
        bson = types.ModuleType("bson")
        bson._uc_fake = True
        sys.modules["bson"] = bson
    int64_mod = types.ModuleType("bson.int64")
    int64_mod.Int64 = FakeInt64
    objectid_mod = types.ModuleType("bson.objectid")
    objectid_mod.ObjectId = FakeObjectId
    sys.modules["bson.int64"] = int64_mod
    sys.modules["bson.objectid"] = objectid_mod
    bson.int64 = int64_mod
    bson.objectid = objectid_mod


def _xyz(v) -> dict:
    return {"X": float(v[0]), "Y": float(v[1]), "Z": float(v[2])}


def make_player(player_id: int, rating: int | None = None) -> dict:
    p = {"StaticInfo": {"PlayerID": int(player_id)}}
    if rating is not None:
        p["PremierRatingNew"] = int(rating)
    return p


def make_flashed(player_id: int, team: int, duration: float, killed_by: int | None = None,
                 killed_tick: int = 0, pos=(0, 0, 0)) -> dict:
    v = {"PlayerID": int(player_id), "FlashDuration": float(duration),
         "FlashDurationAdded": float(duration), "ExistingFlashDuration": 0.0,
         "Position": _xyz(pos), "ViewAngle": {"X": 0.0, "Y": 0.0}, "Team": int(team)}
    if killed_by is not None:
        v["WasKilled"] = {"FrameNumber": int(killed_tick) // 2, "TickNumber": int(killed_tick),
                          "By": int(killed_by), "From": _xyz((0, 0, 0)),
                          "RemainingFlashDuration": 0.5}
    return v


def make_damaged(player_id: int, team: int, damages: list[tuple[int, int, bool]]) -> dict:
    """damages: [(health, armor, was_killed), ...] — по одной записи Damage[] на тик урона."""
    out = []
    for i, (hp, armor, killed) in enumerate(damages):
        d = {"FrameNumber": 500 + i, "TickNumber": 1000 + i, "HealthDamage": int(hp),
             "ArmorDamage": int(armor), "Position": _xyz((0, 0, 0))}
        if killed:
            d["WasKilled"] = True
        out.append(d)
    return {"PlayerID": int(player_id), "Team": int(team), "Damage": out}


def make_grenade(thrower_id, team, round=1, start_tick=1000, end_tick=1100, pos=(0, 0, 0),
                 to=(100, 0, 0), yaw=0.0, pitch=0.0, crouch=False, airborne=False, vel=(0, 0, 0),
                 projectile_id=1, **extra) -> dict:
    general = {
        "RoundIndexDemo": int(round),
        "RoundIndexMatch": int(round),
        "ProjectileID": FakeInt64(projectile_id),
        "ThrowerID": int(thrower_id),
        "Team": int(team),
        "ThrowPoint": {"Position": _xyz(pos), "Velocity": _xyz(vel), "ViewX": float(yaw),
                       "ViewY": float(pitch), "IsCrouching": bool(crouch),
                       "IsAirborne": bool(airborne)},
        "StartProperVelocity": _xyz(vel),
        "StartFrame": int(start_tick) // 2,
        "StartTick": int(start_tick),
        "EndFrame": int(end_tick) // 2,
        "EndTick": int(end_tick),
        "PosFrom": _xyz((pos[0], pos[1], pos[2] + 64)),
        "PosTo": _xyz(to),
    }
    g = {"GeneralInfo": general}
    g.update(copy.deepcopy(extra))
    return g


def make_match(match_id: int, flashes: list[dict] = (), hes: list[dict] = (), fires: list[dict] = (),
               smokes: list[dict] = (), players: list[dict] | None = None) -> dict:
    grenades = [*smokes, *flashes, *hes, *fires]
    if players is None:
        ids = sorted({g["GeneralInfo"]["ThrowerID"] for g in grenades})
        players = [make_player(pid) for pid in ids]
    return {
        "_id": FakeObjectId(int(match_id).to_bytes(12, "big")),
        "MatchID": FakeInt64(match_id),
        "Players": copy.deepcopy(list(players)),
        "Fires": copy.deepcopy(list(fires)),
        "Smokes": copy.deepcopy(list(smokes)),
        "HEs": copy.deepcopy(list(hes)),
        "Flashbangs": copy.deepcopy(list(flashes)),
    }


def write_fixture_pickle(path: Path, matches: list[dict]) -> None:
    _register_fake_bson()
    # Глубокая копия по матчу: между матчами не остаётся общих объектов (как в реальном файле).
    data = [copy.deepcopy(m) for m in matches]
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=4)
