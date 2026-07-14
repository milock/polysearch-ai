"""Tests for polysearch.ratelimit — cross-process token-bucket limiter.

All tests are hermetic: a fake clock + fake sleep function are injected so
no test performs a real wall-clock sleep, and every ``RateLimiter`` is pointed
at a ``tmp_path`` state directory so nothing touches the real ``~/.cache``.
Cross-process coordination is exercised by pointing two independent
``RateLimiter`` instances at the same ``tmp_path`` state directory (simulating
two OS processes) while sharing one fake clock, so the file-ledger handoff is
what's actually under test.
"""

from __future__ import annotations

import contextlib
import json
import logging

import pytest

from polysearch import ratelimit
from polysearch.ratelimit import RateLimiter


class FakeClock:
    def __init__(self, start: float = 1_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


def _fake_sleep_factory(clock: FakeClock):
    calls: list[float] = []

    async def _sleep(seconds: float) -> None:
        calls.append(seconds)
        clock.now += seconds

    _sleep.calls = calls  # type: ignore[attr-defined]
    return _sleep


def _make_limiter(state_dir, clock=None, sleep=None) -> RateLimiter:
    clock = clock or FakeClock()
    sleep = sleep or _fake_sleep_factory(clock)
    return RateLimiter(state_dir=state_dir, clock=clock, sleep=sleep)


# -----------------------------------------------------------------------------
# Burst behavior
# -----------------------------------------------------------------------------


async def test_burst_n_plus_one_delays_last_acquire(tmp_path):
    clock = FakeClock()
    sleep = _fake_sleep_factory(clock)
    limiter = _make_limiter(tmp_path, clock=clock, sleep=sleep)
    n = 5

    for _ in range(n):
        async with limiter.acquire("perplexity", rpm=n):
            pass
    assert sleep.calls == []  # first N proceed with no delay

    async with limiter.acquire("perplexity", rpm=n):
        pass
    assert len(sleep.calls) == 1
    assert sleep.calls[0] > 0


# -----------------------------------------------------------------------------
# Cross-process coordination (two instances, same state dir)
# -----------------------------------------------------------------------------


async def test_two_instances_share_one_provider_budget(tmp_path):
    clock = FakeClock()
    sleep = _fake_sleep_factory(clock)
    limiter_a = _make_limiter(tmp_path, clock=clock, sleep=sleep)
    limiter_b = _make_limiter(tmp_path, clock=clock, sleep=sleep)
    n = 3

    for _ in range(n):
        async with limiter_a.acquire("firecrawl", rpm=n):
            pass

    # Budget for "firecrawl" is exhausted from limiter_a's writes — limiter_b
    # must see that via the shared state file and wait, not proceed instantly.
    assert sleep.calls == []
    async with limiter_b.acquire("firecrawl", rpm=n):
        pass
    assert len(sleep.calls) == 1
    assert sleep.calls[0] > 0


async def test_record_429_pushes_next_slot_across_instances(tmp_path):
    clock = FakeClock()
    sleep = _fake_sleep_factory(clock)
    limiter_a = _make_limiter(tmp_path, clock=clock, sleep=sleep)
    limiter_b = _make_limiter(tmp_path, clock=clock, sleep=sleep)

    limiter_a.record_429("perplexity", retry_after=30.0)

    start = clock.now
    async with limiter_b.acquire("perplexity", rpm=50):
        pass
    assert clock.now >= start + 30.0
    assert sleep.calls and sleep.calls[0] >= 30.0


async def test_record_429_default_backoff_when_no_retry_after(tmp_path):
    clock = FakeClock()
    sleep = _fake_sleep_factory(clock)
    limiter_a = _make_limiter(tmp_path, clock=clock, sleep=sleep)
    limiter_b = _make_limiter(tmp_path, clock=clock, sleep=sleep)

    limiter_a.record_429("firecrawl", retry_after=None)

    start = clock.now
    async with limiter_b.acquire("firecrawl", rpm=50):
        pass
    assert clock.now > start  # some default backoff was applied


# -----------------------------------------------------------------------------
# Corrupt state
# -----------------------------------------------------------------------------


async def test_corrupt_state_file_resets_with_warning(tmp_path, caplog):
    provider = "brave"
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / f"{provider}.json").write_text("{not valid json")

    limiter = _make_limiter(tmp_path)
    with caplog.at_level(logging.WARNING):
        async with limiter.acquire(provider, rpm=10):
            pass

    assert any("ratelimit" in r.message.lower() for r in caplog.records)
    data = json.loads((tmp_path / f"{provider}.json").read_text())
    assert isinstance(data, dict)
    assert data["timestamps"]  # the acquire that just ran got recorded


# -----------------------------------------------------------------------------
# Lock-timeout degrade (cross-process state unavailable mid-run)
# -----------------------------------------------------------------------------


async def test_lock_timeout_degrades_to_in_process_window(tmp_path, caplog):
    clock = FakeClock()
    sleep = _fake_sleep_factory(clock)
    limiter = _make_limiter(tmp_path, clock=clock, sleep=sleep)

    @contextlib.contextmanager
    def _boom(provider):
        raise TimeoutError("simulated lock timeout")
        yield  # pragma: no cover — generator body never runs, raise fires on enter

    limiter._locked_state = _boom  # simulates every lock acquisition timing out

    n = 3
    with caplog.at_level(logging.WARNING):
        for _ in range(n):
            async with limiter.acquire("perplexity", rpm=n):
                pass
        assert sleep.calls == []  # first N still pace correctly via the local fallback

        async with limiter.acquire("perplexity", rpm=n):
            pass  # the N+1th must still wait — the degrade path paces too, not just no-ops

    assert len(sleep.calls) == 1
    assert sleep.calls[0] > 0
    degrade_warnings = [r for r in caplog.records if "degrading to in-process-only" in r.message]
    assert len(degrade_warnings) == 1  # warned once, not once per acquire call


# -----------------------------------------------------------------------------
# Env override (POLYSEARCH_RPM_<PROVIDER>)
# -----------------------------------------------------------------------------


def test_env_override_respected(monkeypatch):
    monkeypatch.setenv("POLYSEARCH_RPM_PERPLEXITY", "5")
    assert ratelimit._rpm_for("perplexity") == 5.0


def test_github_search_rpm_bumped_with_token(monkeypatch):
    monkeypatch.delenv("POLYSEARCH_RPM_GITHUB_SEARCH", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    assert ratelimit._rpm_for("github_search") == 10.0
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    assert ratelimit._rpm_for("github_search") == 30.0


def test_explicit_rpm_override_wins_over_env(monkeypatch):
    monkeypatch.setenv("POLYSEARCH_RPM_PERPLEXITY", "5")
    assert ratelimit._rpm_for("perplexity", 99) == 99


def test_invalid_env_override_is_ignored(monkeypatch):
    monkeypatch.setenv("POLYSEARCH_RPM_PERPLEXITY", "not-a-number")
    assert ratelimit._rpm_for("perplexity") == ratelimit.DEFAULT_RPM["perplexity"]


# -----------------------------------------------------------------------------
# Provider table (public defaults per the release brief)
# -----------------------------------------------------------------------------


def test_default_provider_table_matches_release_spec():
    assert ratelimit.DEFAULT_RPM["perplexity"] == 50
    assert ratelimit.DEFAULT_RPM["firecrawl"] == 100
    assert ratelimit.DEFAULT_RPM["brave"] == 60  # 1 rps free tier
    assert ratelimit.DEFAULT_RPM["scrapecreators"] == 60
    assert ratelimit.DEFAULT_RPM["reddit"] == 90
    assert ratelimit.DEFAULT_RPM["github_search"] == 10
    assert ratelimit.DEFAULT_RPM["hn"] == 150
    assert ratelimit.DEFAULT_RPM["bluesky"] == 30
    assert ratelimit.DEFAULT_RPM["youtube"] is None  # quota-limited, not rate-limited


# -----------------------------------------------------------------------------
# Youtube exemption
# -----------------------------------------------------------------------------


async def test_youtube_is_exempt_never_sleeps(tmp_path):
    clock = FakeClock()

    async def _boom_sleep(_seconds: float) -> None:
        raise AssertionError("exempt provider should never sleep")

    limiter = _make_limiter(tmp_path, clock=clock, sleep=_boom_sleep)
    for _ in range(50):
        async with limiter.acquire("youtube"):
            pass


# -----------------------------------------------------------------------------
# Sync context manager (for sync call sites)
# -----------------------------------------------------------------------------


def test_acquire_sync_delays_burst(tmp_path):
    clock = FakeClock(0.0)
    sleeps: list[float] = []

    def _sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock.now += seconds

    limiter = RateLimiter(state_dir=tmp_path, clock=clock, sleep_sync=_sleep)
    n = 2
    for _ in range(n):
        with limiter.acquire_sync("scrapecreators", rpm=n):
            pass
    assert sleeps == []
    with limiter.acquire_sync("scrapecreators", rpm=n):
        pass
    assert len(sleeps) == 1
    assert sleeps[0] > 0


# -----------------------------------------------------------------------------
# Module-level convenience wrappers delegate to a default instance
# -----------------------------------------------------------------------------


async def test_module_level_acquire_and_record_429(tmp_path, monkeypatch):
    monkeypatch.setattr(ratelimit, "_default_limiter", None)
    monkeypatch.setattr(ratelimit, "_default_state_dir", lambda: tmp_path)
    async with ratelimit.acquire("perplexity", rpm=100):
        pass
    ratelimit.record_429("perplexity", retry_after=1.0)
    state = json.loads((tmp_path / "perplexity.json").read_text())
    assert state["blocked_until"] > 0


def test_default_state_dir_is_under_polysearch_cache():
    # The ledger lives under ~/.cache/polysearch/ratelimit/.
    path = ratelimit._default_state_dir()
    assert path.parts[-2:] == ("polysearch", "ratelimit")


# -----------------------------------------------------------------------------
# Integration: representative call sites actually route through the limiter.
# -----------------------------------------------------------------------------


def _patch_recording_acquire(monkeypatch: pytest.MonkeyPatch, calls: list[str]) -> None:
    """Replace ``ratelimit.acquire`` (the shared module attribute every call
    site references) with a no-op that records the provider name it was
    invoked with."""

    @contextlib.asynccontextmanager
    async def _fake_acquire(provider: str, *, rpm: float | None = None):
        calls.append(provider)
        yield

    monkeypatch.setattr(ratelimit, "acquire", _fake_acquire)


async def test_perplexity_run_one_acquires_perplexity_provider(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    from polysearch.providers import perplexity as pp

    calls: list[str] = []
    _patch_recording_acquire(monkeypatch, calls)

    mock_resp = MagicMock()
    mock_resp.model_dump.return_value = {
        "choices": [{"message": {"content": "ok"}}],
        "citations": [],
        "search_results": [],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=mock_resp)

    result = await pp._run_one(
        client,
        "q",
        model="sonar-pro",
        recency=None,
        domain_filter=None,
        search_mode="web",
        context_size="low",
    )
    assert result.error is None
    assert calls == ["perplexity"]


async def test_linkedin_enrich_acquires_scrapecreators_provider(monkeypatch):
    import httpx
    import respx

    from polysearch.config import Settings
    from polysearch.providers.linkedin import LinkedInEnricher

    calls: list[str] = []
    _patch_recording_acquire(monkeypatch, calls)

    with respx.mock:
        respx.get("https://api.scrapecreators.com/v1/linkedin/profile").mock(
            return_value=httpx.Response(200, json={"success": True, "name": "Jane"})
        )
        enricher = LinkedInEnricher(Settings(scrapecreators_api_key="sc-test"))
        result = await enricher.enrich("https://www.linkedin.com/in/jane/")

    assert result is not None
    assert calls == ["scrapecreators"]


async def test_provider_429_broadcasts_backoff_to_shared_ledger(tmp_path, monkeypatch):
    """A live 429 with Retry-After, seen by one process's provider call, must
    push the next slot out for a *second* limiter instance sharing the ledger."""
    import httpx
    import respx

    from polysearch.config import Settings
    from polysearch.providers.linkedin import LinkedInEnricher

    clock = FakeClock()
    sleep = _fake_sleep_factory(clock)
    # The default limiter (what the provider's record_429 writes into) shares
    # the ledger dir + clock with limiter_b below — simulating two processes.
    default = RateLimiter(state_dir=tmp_path, clock=clock, sleep=sleep)
    monkeypatch.setattr(ratelimit, "_default_limiter", default)

    with respx.mock:
        respx.get("https://api.scrapecreators.com/v1/linkedin/profile").mock(
            return_value=httpx.Response(429, headers={"Retry-After": "30"}, json={})
        )
        result = await LinkedInEnricher(Settings(scrapecreators_api_key="k")).enrich(
            "https://www.linkedin.com/in/x/"
        )
    assert result is None  # the 429 surfaced as a failure

    limiter_b = RateLimiter(state_dir=tmp_path, clock=clock, sleep=sleep)
    start = clock.now
    async with limiter_b.acquire("scrapecreators", rpm=60):
        pass
    assert clock.now >= start + 30.0
    assert sleep.calls and sleep.calls[-1] >= 30.0


async def test_perplexity_run_one_records_429_on_rate_limit(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    from tenacity import wait_none

    from polysearch.providers import perplexity as pp

    # Zero out tenacity backoff so the 3 retry attempts run instantly.
    monkeypatch.setattr(pp, "wait_exponential_jitter", lambda **_kw: wait_none())

    recorded: list[tuple[str, float | None]] = []
    monkeypatch.setattr(
        ratelimit,
        "record_429",
        lambda provider, retry_after=None: recorded.append((provider, retry_after)),
    )

    class _RateLimited(Exception):
        status_code = 429

        def __init__(self) -> None:
            super().__init__("429 rate limit")
            self.response = SimpleNamespace(headers={"retry-after": "12"})

    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=_RateLimited())

    result = await pp._run_one(
        client,
        "q",
        model="sonar-pro",
        recency=None,
        domain_filter=None,
        search_mode="web",
        context_size="low",
    )
    assert result.error is not None  # exhausted retries → structured error
    assert recorded  # at least one 429 broadcast
    assert recorded[0] == ("perplexity", 12.0)
