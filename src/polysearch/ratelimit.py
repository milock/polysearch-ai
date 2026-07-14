"""Shared cross-process token-bucket rate limiter for external API calls.

Motivation: concurrent research runs and sub-agents may invoke polysearch
independently, but they share the same upstream API accounts (Perplexity,
Firecrawl, etc). An in-process-only limiter can't see what a sibling process
is doing, so the shared account budget gets blown even when each process
individually paces itself.

Coordination: each provider gets a sliding 60s window of request timestamps
persisted to ``~/.cache/polysearch/ratelimit/<provider>.json``, guarded by
``fcntl.flock`` on a sidecar ``.lock`` file so every process reads-prunes-
appends-writes atomically. ``record_429()`` additionally stamps a
``blocked_until`` floor into the same file so a 429 seen by one process backs
off every process sharing that provider.

Usage::

    from polysearch import ratelimit

    async with ratelimit.acquire("perplexity"):
        resp = await client.chat.completions.create(...)

    # on a 429:
    ratelimit.record_429("perplexity", retry_after=retry_after_header)

Never crashes the caller: cache-dir creation failures or lock timeouts degrade
to in-process-only limiting (logged once); a corrupt state file is reset in
place (logged) and the acquire proceeds normally.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import json
import logging
import os
import time
from pathlib import Path
from typing import AsyncIterator, Callable, Iterator

log = logging.getLogger(__name__)

WINDOW_SECONDS = 60.0

# Module-level defaults (requests per minute). ``None`` marks a provider as
# exempt from rate limiting entirely (quota-based, not rate-based).
DEFAULT_RPM: dict[str, float | None] = {
    "perplexity": 50,
    "firecrawl": 100,
    "brave": 60,  # 1 rps free tier
    "scrapecreators": 60,
    "reddit": 90,
    "github_search": 10,  # 30 with GITHUB_TOKEN set — see _rpm_for
    "hn": 150,
    "bluesky": 30,
    "youtube": None,  # quota-limited by Google, not rate-limited here
}

_EXEMPT_PROVIDERS = {p for p, rpm in DEFAULT_RPM.items() if rpm is None}

_DEFAULT_BACKOFF_SECONDS = 5.0  # used by record_429 when no Retry-After given
_LOCK_TIMEOUT_SECONDS = 2.0
_LOCK_POLL_SECONDS = 0.05


def _default_state_dir() -> Path:
    return Path.home() / ".cache" / "polysearch" / "ratelimit"


def _rpm_for(provider: str, rpm_override: float | None = None) -> float | None:
    """Resolve the effective rpm for a provider.

    Precedence: explicit override > ``POLYSEARCH_RPM_<PROVIDER>`` env var >
    GITHUB_TOKEN bump (github_search only) > module default.
    """
    if rpm_override is not None:
        return rpm_override
    env_val = os.environ.get(f"POLYSEARCH_RPM_{provider.upper()}")
    if env_val:
        try:
            return float(env_val)
        except ValueError:
            log.warning(
                "ratelimit: ignoring invalid %s=%r",
                f"POLYSEARCH_RPM_{provider.upper()}",
                env_val,
            )
    if provider == "github_search" and os.environ.get("GITHUB_TOKEN"):
        return 30.0
    return DEFAULT_RPM.get(provider, 60.0)


class RateLimiter:
    """Token-bucket limiter with cross-process coordination via a file ledger.

    ``clock`` / ``sleep`` / ``sleep_sync`` are injectable so tests never perform
    a real wall-clock sleep.
    """

    def __init__(
        self,
        *,
        state_dir: Path | str | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], "asyncio.Future[None]"] = asyncio.sleep,
        sleep_sync: Callable[[float], None] = time.sleep,
    ) -> None:
        self._state_dir = Path(state_dir) if state_dir is not None else _default_state_dir()
        self._clock = clock
        self._sleep = sleep
        self._sleep_sync = sleep_sync
        # In-process-only fallback state, used only when the file ledger is
        # unavailable (cache dir can't be created, or the lock times out).
        self._local_timestamps: dict[str, list[float]] = {}
        self._local_blocked_until: dict[str, float] = {}
        self._warned_degraded: set[str] = set()

    # -- paths -----------------------------------------------------------

    def _paths(self, provider: str) -> tuple[Path, Path]:
        return (
            self._state_dir / f"{provider}.json",
            self._state_dir / f"{provider}.lock",
        )

    # -- state IO (must be called while holding the flock) ----------------

    def _load_state(self, provider: str, state_path: Path) -> dict:
        try:
            raw = state_path.read_text()
        except FileNotFoundError:
            return {"timestamps": [], "blocked_until": 0.0}
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("state file did not contain a JSON object")
        except (json.JSONDecodeError, ValueError) as exc:
            log.warning(
                "ratelimit: corrupt state file for provider %r (%s) — resetting", provider, exc
            )
            return {"timestamps": [], "blocked_until": 0.0}
        data.setdefault("timestamps", [])
        data.setdefault("blocked_until", 0.0)
        return data

    def _write_state(self, state_path: Path, state: dict) -> None:
        tmp_path = state_path.with_suffix(state_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(state))
        os.replace(tmp_path, state_path)

    @contextlib.contextmanager
    def _locked_state(self, provider: str) -> Iterator[tuple[dict, Path]]:
        """Open + flock the provider's ledger, yield (state, state_path).

        Raises OSError/TimeoutError on any failure to acquire the file lock or
        create the cache dir — callers degrade to in-process-only limiting.
        """
        state_path, lock_path = self._paths(provider)
        self._state_dir.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "a+") as lockf:
            deadline = time.time() + _LOCK_TIMEOUT_SECONDS
            while True:
                try:
                    fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.time() >= deadline:
                        raise TimeoutError(f"ratelimit: lock timeout for provider {provider!r}")
                    time.sleep(_LOCK_POLL_SECONDS)
            try:
                state = self._load_state(provider, state_path)
                yield state, state_path
            finally:
                fcntl.flock(lockf, fcntl.LOCK_UN)

    def _warn_degraded(self, provider: str, exc: Exception) -> None:
        if provider not in self._warned_degraded:
            self._warned_degraded.add(provider)
            log.warning(
                "ratelimit: cross-process state unavailable for provider %r (%s) — "
                "degrading to in-process-only limiting",
                provider,
                exc,
            )

    # -- reservation --------------------------------------------------------

    def _try_reserve(self, provider: str, limit: float | None) -> float:
        """Attempt to reserve a slot now. Returns 0.0 if reserved, else the
        number of seconds the caller should wait before retrying."""
        now = self._clock()
        try:
            with self._locked_state(provider) as (state, state_path):
                timestamps = [t for t in state["timestamps"] if now - t < WINDOW_SECONDS]
                blocked_until = float(state.get("blocked_until") or 0.0)
                if blocked_until > now:
                    return blocked_until - now
                if limit is None or len(timestamps) < limit:
                    timestamps.append(now)
                    state["timestamps"] = timestamps
                    state["blocked_until"] = blocked_until
                    self._write_state(state_path, state)
                    return 0.0
                return max(timestamps[0] + WINDOW_SECONDS - now, 0.01)
        except (OSError, TimeoutError) as exc:
            self._warn_degraded(provider, exc)
            return self._try_reserve_local(provider, limit, now)

    def _try_reserve_local(self, provider: str, limit: float | None, now: float) -> float:
        blocked_until = self._local_blocked_until.get(provider, 0.0)
        if blocked_until > now:
            return blocked_until - now
        timestamps = [t for t in self._local_timestamps.get(provider, []) if now - t < WINDOW_SECONDS]
        if limit is None or len(timestamps) < limit:
            timestamps.append(now)
            self._local_timestamps[provider] = timestamps
            return 0.0
        self._local_timestamps[provider] = timestamps
        return max(timestamps[0] + WINDOW_SECONDS - now, 0.01)

    # -- public API -----------------------------------------------------

    @contextlib.asynccontextmanager
    async def acquire(self, provider: str, *, rpm: float | None = None) -> AsyncIterator[None]:
        if provider in _EXEMPT_PROVIDERS:
            yield
            return
        limit = _rpm_for(provider, rpm)
        while True:
            # Off-loaded to a thread: _try_reserve does blocking file IO and,
            # under real cross-process lock contention, a blocking time.sleep
            # poll — neither may run directly on the event loop.
            wait = await asyncio.to_thread(self._try_reserve, provider, limit)
            if wait <= 0:
                break
            await self._sleep(wait)
        yield

    @contextlib.contextmanager
    def acquire_sync(self, provider: str, *, rpm: float | None = None) -> Iterator[None]:
        if provider in _EXEMPT_PROVIDERS:
            yield
            return
        limit = _rpm_for(provider, rpm)
        while True:
            wait = self._try_reserve(provider, limit)
            if wait <= 0:
                break
            self._sleep_sync(wait)
        yield

    def record_429(self, provider: str, retry_after: float | None = None) -> None:
        """Push the provider's next-available slot out, visible to every process
        sharing this state dir. Uses ``retry_after`` seconds if given (e.g. from
        a Retry-After header), else a default backoff."""
        if provider in _EXEMPT_PROVIDERS:
            return
        delay = retry_after if retry_after and retry_after > 0 else _DEFAULT_BACKOFF_SECONDS
        now = self._clock()
        blocked_until = now + delay
        try:
            with self._locked_state(provider) as (state, state_path):
                state["blocked_until"] = max(float(state.get("blocked_until") or 0.0), blocked_until)
                self._write_state(state_path, state)
        except (OSError, TimeoutError) as exc:
            self._warn_degraded(provider, exc)
            self._local_blocked_until[provider] = max(
                self._local_blocked_until.get(provider, 0.0), blocked_until
            )


# -----------------------------------------------------------------------------
# Module-level default instance
# -----------------------------------------------------------------------------

_default_limiter: RateLimiter | None = None


def _get_default() -> RateLimiter:
    global _default_limiter
    if _default_limiter is None:
        _default_limiter = RateLimiter(state_dir=_default_state_dir())
    return _default_limiter


def acquire(provider: str, *, rpm: float | None = None):
    """``async with acquire("perplexity"): ...`` — see ``RateLimiter.acquire``."""
    return _get_default().acquire(provider, rpm=rpm)


def acquire_sync(provider: str, *, rpm: float | None = None):
    """``with acquire_sync("scrapecreators"): ...`` for sync call sites."""
    return _get_default().acquire_sync(provider, rpm=rpm)


def record_429(provider: str, retry_after: float | None = None) -> None:
    _get_default().record_429(provider, retry_after)


__all__ = [
    "WINDOW_SECONDS",
    "DEFAULT_RPM",
    "RateLimiter",
    "acquire",
    "acquire_sync",
    "record_429",
]
