"""Безопасный потоковый читатель огромного pickle, верхний уровень которого — список.

Разрешены только два глобала: bson.int64.Int64 и bson.objectid.ObjectId. Их заменяют
локальные заглушки, поэтому произвольный код из файла выполниться не может. Элементы
верхнего списка (матчи) по одному отдаются в callback и после этого освобождаются.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Callable


class Int64(int):
    """Заглушка для bson.int64.Int64: обычный int."""


class ObjectId:
    """Заглушка для bson.objectid.ObjectId: хранит 12 байт идентификатора."""

    __slots__ = ("oid",)

    def __setstate__(self, state):
        if isinstance(state, dict):
            state = state.get("_ObjectId__id", state)
        elif isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], dict):
            state = state[1].get("_ObjectId__id", state)
        self.oid = state

    def __repr__(self):
        oid = getattr(self, "oid", None)
        return f"ObjectId('{oid.hex() if isinstance(oid, bytes) else oid}')"


ALLOWED = {("bson.int64", "Int64"): Int64, ("bson.objectid", "ObjectId"): ObjectId}


class Stop(Exception):
    """callback бросает Stop, чтобы прервать чтение; stream_matches его глотает."""


class _StreamUnpickler(pickle._Unpickler):
    dispatch = dict(pickle._Unpickler.dispatch)

    def __init__(self, f, callback: Callable[[dict], None]):
        super().__init__(f)
        self.callback = callback
        self.top = None

    def find_class(self, module, name):
        try:
            return ALLOWED[(module, name)]
        except KeyError:
            raise pickle.UnpicklingError(f"blocked global {module}.{name}") from None

    def persistent_load(self, pid):
        raise pickle.UnpicklingError("persistent ids are not allowed")

    def _flush(self, items):
        for it in items:
            self.callback(it)
        # Убираем из memo ссылки на составные объекты, чтобы отданные матчи освободились.
        memo = self.memo
        for k in [k for k, v in memo.items()
                  if v is not self.top and not isinstance(v, (str, bytes, int, float, type))]:
            del memo[k]

    def load_empty_list(self):
        lst = []
        if self.top is None:
            self.top = lst
        self.append(lst)
    dispatch[pickle.EMPTY_LIST[0]] = load_empty_list

    def load_appends(self):
        items = self.pop_mark()
        lst = self.stack[-1]
        if lst is self.top:
            self._flush(items)
        else:
            lst.extend(items)
    dispatch[pickle.APPENDS[0]] = load_appends

    def load_append(self):
        value = self.stack.pop()
        lst = self.stack[-1]
        if lst is self.top:
            self._flush([value])
        else:
            lst.append(value)
    dispatch[pickle.APPEND[0]] = load_append


def stream_matches(path: Path, callback: Callable[[dict], None]) -> None:
    """Читает pickle по пути path и вызывает callback(match) для каждого элемента верхнего списка."""
    with open(path, "rb", buffering=16 * 1024 * 1024) as f:
        try:
            _StreamUnpickler(f, callback).load()
        except Stop:
            pass
