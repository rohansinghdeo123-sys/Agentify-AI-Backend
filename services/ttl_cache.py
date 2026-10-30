"""Tiny thread-safe in-process TTL cache.

Used to absorb hot, repeated reads (leaderboard Firebase lookups, admin
console polling). Per-process by design: entries are short-lived enough that
multi-worker divergence is acceptable, and no new infrastructure is needed.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, Hashable, Tuple


class TTLCache:
    def __init__(self, max_entries: int = 256, clock: Callable[[], float] = time.monotonic) -> None:
        self._max_entries = max_entries
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: Dict[Hashable, Tuple[float, Any]] = {}
        self._pending: Dict[Hashable, threading.Event] = {}
        self._generation = 0

    def get_or_build(self, key: Hashable, ttl_seconds: float, builder: Callable[[], Any]) -> Any:
        while True:
            with self._lock:
                now = self._clock()
                entry = self._entries.get(key)
                if entry is not None and entry[0] > now:
                    return entry[1]
                pending = self._pending.get(key)
                if pending is None:
                    pending = threading.Event()
                    self._pending[key] = pending
                    generation = self._generation
                    break
            pending.wait()
        try:
            value = builder()
            with self._lock:
                now = self._clock()
                if generation == self._generation:
                    self._evict_expired_locked(now)
                    if len(self._entries) >= self._max_entries:
                        oldest_key = min(self._entries, key=lambda item: self._entries[item][0])
                        self._entries.pop(oldest_key, None)
                    self._entries[key] = (now + ttl_seconds, value)
            return value
        finally:
            with self._lock:
                self._pending.pop(key, None)
                pending.set()

    def invalidate(self, key: Hashable) -> None:
        with self._lock:
            self._generation += 1
            self._entries.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._generation += 1
            self._entries.clear()

    def _evict_expired_locked(self, now: float) -> None:
        expired = [key for key, (expires_at, _) in self._entries.items() if expires_at <= now]
        for key in expired:
            self._entries.pop(key, None)
