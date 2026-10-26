import os
import pickle
import pickletools

import pytest

from tests.fixtures import make_grenade, make_match, write_fixture_pickle
from utility_coach import safe_pickle
from utility_coach.safe_pickle import Int64, ObjectId, Stop, stream_matches

CALLED = []


def _boom(*args):
    CALLED.append(args)
    return "boom"


class _Evil:
    def __init__(self, func, args):
        self.func, self.args = func, args

    def __reduce__(self):
        return (self.func, self.args)


def _small_match(i: int) -> dict:
    fl = make_grenade(100 + i % 10, 2, round=i % 30, projectile_id=10_000 + i)
    return make_match(i, flashes=[fl])


def _read_all(path):
    got = []
    stream_matches(path, got.append)
    return got


def _opcodes(path):
    with open(path, "rb") as f:
        return [op.name for op, _, _ in pickletools.genops(f)]


def test_int64_and_objectid_restored(tmp_path):
    path = tmp_path / "one.list"
    m = make_match(7_000_000_001, flashes=[make_grenade(5, 3, projectile_id=2**62 + 3)])
    write_fixture_pickle(path, [m])
    ops = _opcodes(path)
    assert "STACK_GLOBAL" in ops and "APPEND" in ops  # один матч -> верхний APPEND

    (got,) = _read_all(path)
    assert type(got["MatchID"]) is Int64 and got["MatchID"] == 7_000_000_001
    assert isinstance(got["_id"], ObjectId)
    assert got["_id"].oid == (7_000_000_001).to_bytes(12, "big")
    pid = got["Flashbangs"][0]["GeneralInfo"]["ProjectileID"]
    assert type(pid) is Int64 and pid == 2**62 + 3
    assert got["Flashbangs"][0]["GeneralInfo"]["ThrowPoint"]["Position"] == {"X": 0.0, "Y": 0.0, "Z": 0.0}


def test_appends_batches_over_1000_matches(tmp_path):
    path = tmp_path / "many.list"
    write_fixture_pickle(path, [_small_match(i) for i in range(1005)])
    assert _opcodes(path).count("APPENDS") >= 2  # pickle пишет пачками по 1000

    got = _read_all(path)
    assert [int(m["MatchID"]) for m in got] == list(range(1005))
    last = got[-1]["Flashbangs"][0]["GeneralInfo"]
    assert last["ThrowerID"] == 100 + 1004 % 10
    assert int(last["ProjectileID"]) == 10_000 + 1004
    assert last["RoundIndexMatch"] == 1004 % 30


def test_stop_interrupts_reading(tmp_path):
    path = tmp_path / "many.list"
    write_fixture_pickle(path, [_small_match(i) for i in range(1005)])
    got = []

    def cb(m):
        got.append(int(m["MatchID"]))
        if len(got) == 3:
            raise Stop

    stream_matches(path, cb)  # Stop не выходит наружу
    assert got == [0, 1, 2]


def test_other_callback_errors_propagate(tmp_path):
    path = tmp_path / "one.list"
    write_fixture_pickle(path, [_small_match(0)])

    def cb(m):
        raise ValueError("bad")

    with pytest.raises(ValueError):
        stream_matches(path, cb)


def test_foreign_global_os_system_blocked(tmp_path):
    path = tmp_path / "evil.list"
    path.write_bytes(pickle.dumps([{"x": _Evil(os.system, ("echo pwned",))}], protocol=4))
    with pytest.raises(pickle.UnpicklingError, match="blocked global"):
        stream_matches(path, lambda m: None)


def test_foreign_global_not_executed(tmp_path):
    path = tmp_path / "evil2.list"
    path.write_bytes(pickle.dumps([_Evil(_boom, ("x",))], protocol=4))
    CALLED.clear()
    with pytest.raises(pickle.UnpicklingError):
        stream_matches(path, lambda m: None)
    assert CALLED == []


def test_protocol0_global_blocked(tmp_path):
    path = tmp_path / "evil0.list"
    path.write_bytes(b"cbuiltins\neval\n(S'1+1'\ntR.")
    with pytest.raises(pickle.UnpicklingError):
        stream_matches(path, lambda m: None)


@pytest.mark.parametrize("state", [
    b"\x01" * 12,
    {"_ObjectId__id": b"\x01" * 12},
    (None, {"_ObjectId__id": b"\x01" * 12}),
])
def test_objectid_setstate_forms(state):
    oid = ObjectId.__new__(ObjectId)
    oid.__setstate__(state)
    assert oid.oid == b"\x01" * 12
    assert repr(oid) == "ObjectId('" + "01" * 12 + "')"


def test_whitelist_is_exactly_two_globals():
    assert set(safe_pickle.ALLOWED) == {("bson.int64", "Int64"), ("bson.objectid", "ObjectId")}
