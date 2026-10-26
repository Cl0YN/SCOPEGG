import math

import pyarrow.parquet as pq
import pytest

from tests.fixtures import (make_damaged, make_flashed, make_grenade, make_match, make_player,
                            write_fixture_pickle)
from utility_coach.ingest import SCHEMAS, flatten_match, run_ingest

T, CT = 2, 3


def _match_a():
    flash = make_grenade(
        11, T, round=3, start_tick=5000, end_tick=5100, pos=(10, 20, 30), to=(400, 50, 35),
        yaw=12.5, pitch=-3.0, crouch=True, vel=(200, 0, 0), projectile_id=2**60 + 1,
        FlashedPlayers=[
            make_flashed(21, CT, 2.5, killed_by=12, killed_tick=5300),
            make_flashed(12, T, 1.5),
            make_flashed(11, T, 0.7),
        ],
    )
    flash_empty = make_grenade(12, T, round=4, projectile_id=2)  # без FlashedPlayers
    he = make_grenade(
        21, CT, round=3, airborne=True, projectile_id=3,
        DamagedPlayers=[
            make_damaged(11, T, [(10, 5, False), (15, 3, False)]),
            make_damaged(12, T, [(40, 0, False), (60, 0, True)]),
        ],
    )
    he_empty = make_grenade(22, CT, round=5, projectile_id=4)  # без DamagedPlayers
    fire_ok = make_grenade(
        11, T, round=6, projectile_id=5,
        FinalInfernoInfo={"UniqueID": 777, "ConvexHull3D": {"Vertices": [], "Indices": None}},
        DamagedPlayers=[make_damaged(21, CT, [(8, 0, False)])],
    )
    fire_dud = make_grenade(12, T, round=6, projectile_id=6, FinalInfernoInfo={"UniqueID": 0})
    smoke = make_grenade(22, CT, round=2, projectile_id=7)
    players = [make_player(11, 15000), make_player(12), make_player(21, 9000), make_player(22)]
    return make_match(1001, flashes=[flash, flash_empty], hes=[he, he_empty],
                      fires=[fire_ok, fire_dud], smokes=[smoke], players=players)


def _match_b():
    fl = make_grenade(31, CT, projectile_id=1, FlashedPlayers=[make_flashed(41, T, 3.0)])
    return make_match(1002, flashes=[fl], players=[make_player(31), make_player(41, 20000)])


def test_flatten_order_and_uid():
    throws, victims, damage, players = flatten_match(_match_a(), uid_start=100)
    assert [r["throw_uid"] for r in throws] == list(range(100, 107))
    assert [r["grenade_type"] for r in throws] == ["smoke", "flash", "flash", "he", "he", "fire", "fire"]
    fl = throws[1]
    assert fl["match_id"] == 1001 and fl["projectile_id"] == 2**60 + 1
    assert (fl["round"], fl["thrower_id"], fl["team"]) == (3, 11, T)
    assert (fl["start_tick"], fl["end_tick"]) == (5000, 5100)
    assert (fl["throw_x"], fl["throw_y"], fl["throw_z"]) == (10.0, 20.0, 30.0)
    assert (fl["to_x"], fl["to_y"], fl["to_z"]) == (400.0, 50.0, 35.0)
    assert (fl["yaw"], fl["pitch"], fl["vel_x"]) == (12.5, -3.0, 200.0)
    assert fl["crouch"] is True and fl["airborne"] is False
    assert throws[3]["airborne"] is True
    assert [r["inferno_ok"] for r in throws] == [True, True, True, True, True, True, False]
    assert len(players) == 4


def test_flash_victims_killed_fields():
    _, victims, _, _ = flatten_match(_match_a(), uid_start=0)
    assert [v["throw_uid"] for v in victims] == [1, 1, 1]
    by_id = {v["victim_id"]: v for v in victims}
    assert by_id[21]["killed"] is True
    assert (by_id[21]["killed_by"], by_id[21]["killed_tick"]) == (12, 5300)
    assert by_id[21]["victim_team"] == CT and by_id[21]["flash_duration"] == 2.5
    for vid in (12, 11):
        assert by_id[vid]["killed"] is False
        assert (by_id[vid]["killed_by"], by_id[vid]["killed_tick"]) == (-1, -1)


def test_damage_summed_per_victim():
    _, _, damage, _ = flatten_match(_match_a(), uid_start=0)
    rows = {(d["throw_uid"], d["victim_id"]): d for d in damage}
    assert set(rows) == {(3, 11), (3, 12), (5, 21)}
    assert (rows[(3, 11)]["health_dmg"], rows[(3, 11)]["armor_dmg"], rows[(3, 11)]["killed"]) == (25, 8, False)
    assert (rows[(3, 12)]["health_dmg"], rows[(3, 12)]["killed"]) == (100, True)
    assert rows[(3, 12)]["grenade_type"] == "he" and rows[(5, 21)]["grenade_type"] == "fire"
    assert rows[(5, 21)]["victim_team"] == CT


def test_players_rating_nan_when_missing():
    *_, players = flatten_match(_match_a(), uid_start=0)
    r = {p["player_id"]: p["rating"] for p in players}
    assert r[11] == 15000.0 and r[21] == 9000.0
    assert math.isnan(r[12]) and math.isnan(r[22])


@pytest.fixture()
def ingested(tmp_path):
    raw = tmp_path / "fx.list"
    write_fixture_pickle(raw, [_match_a(), _match_b()])
    data = tmp_path / "data"
    counts = run_ingest(raw=raw, data=data)
    return data, counts


def test_run_ingest_counts(ingested):
    data, counts = ingested
    assert counts == {"matches": 2, "throws": 8, "flash_victims": 4, "damage_events": 3, "players": 6}
    for name in SCHEMAS:
        assert (data / f"{name}.parquet").exists()
        assert not (data / f"{name}.parquet.tmp").exists()


def test_run_ingest_schemas(ingested):
    data, _ = ingested
    for name, schema in SCHEMAS.items():
        assert pq.read_schema(data / f"{name}.parquet").remove_metadata() == schema
    t = pq.read_table(data / "throws.parquet").schema
    assert str(t.field("round").type) == "int16" and str(t.field("team").type) == "int8"
    assert str(t.field("start_tick").type) == "int32" and str(t.field("to_x").type) == "float"
    assert str(t.field("inferno_ok").type) == "bool"


def test_run_ingest_throw_uid_is_global(ingested):
    data, _ = ingested
    throws = pq.read_table(data / "throws.parquet").to_pandas()
    assert throws["throw_uid"].tolist() == list(range(8))
    assert throws.loc[throws["match_id"] == 1002, "throw_uid"].tolist() == [7]
    victims = pq.read_table(data / "flash_victims.parquet").to_pandas()
    assert victims.loc[victims["match_id"] == 1002, "throw_uid"].tolist() == [7]
    assert set(victims["throw_uid"]) <= set(throws.loc[throws["grenade_type"] == "flash", "throw_uid"])


def test_run_ingest_players_nan(ingested):
    data, _ = ingested
    players = pq.read_table(data / "players.parquet").to_pandas()
    assert players["rating"].isna().sum() == 3
    assert players.loc[players["player_id"] == 41, "rating"].item() == 20000.0


def test_run_ingest_limit(tmp_path):
    raw = tmp_path / "fx.list"
    write_fixture_pickle(raw, [_match_a(), _match_b()])
    counts = run_ingest(raw=raw, data=tmp_path / "data", limit=1)
    assert counts["matches"] == 1 and counts["throws"] == 7 and counts["players"] == 4


def test_run_ingest_rerun_overwrites_not_appends(tmp_path):
    # Review Focus: прогон с --limit, потом полный — parquet должен перезаписаться, а не дописаться
    raw = tmp_path / "fx.list"
    write_fixture_pickle(raw, [_match_a(), _match_b()])
    data = tmp_path / "data"
    run_ingest(raw=raw, data=data, limit=1)
    run_ingest(raw=raw, data=data)
    run_ingest(raw=raw, data=data)
    assert pq.read_metadata(data / "throws.parquet").num_rows == 8
    assert pq.read_metadata(data / "players.parquet").num_rows == 6
