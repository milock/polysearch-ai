"""Native community adapters — Reddit, Hacker News, Bluesky, GitHub, X, YouTube.

Each adapter is a ``CommunitySource``: it exposes a ``name`` and an
``async search(topic, *, window_days=30, limit=20) -> list[SourceResult]`` that
returns tier-``COMMUNITY`` results with ``engagement`` populated and ``layer``
set to the source slug (``reddit``, ``hackernews``, …) so fusion can normalize
per source. Credentials come from ``Settings``, never the process environment.

Every adapter is failure-isolated: any internal error degrades to an empty list
with the cause surfaced on ``self.last_error`` — adapters never raise. Rate
limiting flows through the shared cross-process ``ratelimit`` ledger, and a 429
is broadcast to sibling processes via ``ratelimit.record_429``.

Verified API facts baked in (docs pass, 2026-07-13):
  - Reddit TLS-fingerprint-blocks non-browser HTTP clients at its edge, so the
    official OAuth API (``oauth.reddit.com``) is the default path when
    ``reddit_client_id``/``reddit_client_secret`` are set; unauthenticated
    ``www.reddit.com/search.json`` with a realistic browser User-Agent is the
    fallback; ScrapeCreators' ``/v1/reddit/search`` is the last resort.
  - Hacker News via Algolia (``hn.algolia.com/api/v1/search_by_date``) is
    unauthenticated and stable.
  - Bluesky's public AppView (``public.api.bsky.app``) needs no auth for reads
    but requires a realistic User-Agent — default UAs get 403.
  - GitHub search: 10 req/min unauthenticated, 30/min with ``github_token``.
  - ScrapeCreators documents ``/v1/twitter/user-tweets`` but no keyword-search
    endpoint, so X is handle-based (``x_handles`` watch list, filtered
    client-side).
  - YouTube ``search.list`` has its own 100-searches/day quota bucket — budget
    one call per run.

Adapted from the last30days community-signal library (MIT). See ATTRIBUTION.md.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from datetime import datetime, timedelta, timezone

import httpx

from polysearch import ratelimit
from polysearch.config import Settings
from polysearch.output.schema import SourceResult

log = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 20.0
_TIER = "COMMUNITY"

# Reddit's edge (Fastly) and Bluesky's AppView both fingerprint/block default
# httpx/requests User-Agents; a realistic desktop browser UA gets through both.
_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
_API_UA = "polysearch/1.0"


def _record_429(provider: str, resp: httpx.Response) -> None:
    if resp.status_code == 429:
        ratelimit.record_429(provider, ratelimit.parse_retry_after(resp.headers.get("retry-after")))


def _result(
    *,
    title: str,
    url: str,
    snippet: str,
    date: str | None,
    engagement: int | None,
    source: str,
) -> SourceResult:
    return SourceResult(
        url=url,
        title=title,
        snippet=snippet,
        tier=_TIER,
        published_date=date,
        layer=source,
        engagement=engagement,
    )


# ── date helpers ─────────────────────────────────────────────────────────────


def _since_unix(window_days: int) -> int:
    return int((datetime.now(timezone.utc) - timedelta(days=window_days)).timestamp())


def _since_date_str(window_days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=window_days)).strftime("%Y-%m-%d")


def _since_rfc3339(window_days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=window_days)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _parse_any_date(value: str) -> datetime | None:
    """Best-effort parse of ISO-8601 or legacy Twitter-format dates."""
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        pass
    try:
        return datetime.strptime(value, "%a %b %d %H:%M:%S %z %Y")
    except ValueError:
        return None


def _is_recent(date_value: str | None, window_days: int) -> bool:
    """True if ``date_value`` is unknown (nothing to disqualify it on) or falls
    within the window. Each adapter's defensive post-filter on top of whatever
    server-side windowing the API already applied."""
    if not date_value:
        return True
    dt = _parse_any_date(date_value)
    if dt is None:
        return True
    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
    return dt >= cutoff


# ── Hacker News — Algolia, unauthenticated ──────────────────────────────────

_HN_SEARCH_URL = "https://hn.algolia.com/api/v1/search_by_date"


class HackerNewsSource:
    """Hacker News search via the Algolia API (no auth)."""

    name = "hackernews"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.last_error: str | None = None

    async def search(
        self, topic: str, *, window_days: int = 30, limit: int = 20
    ) -> list[SourceResult]:
        self.last_error = None
        params = {
            "query": topic,
            "tags": "story",
            "numericFilters": f"created_at_i>{_since_unix(window_days)}",
            "hitsPerPage": str(limit),
        }
        # Fetch+parse in one try block — a 200 with an unexpected shape must
        # degrade to [] just like a network/HTTP failure, not raise.
        try:
            async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
                async with ratelimit.acquire("hn"):
                    resp = await client.get(_HN_SEARCH_URL, params=params)
                _record_429("hn", resp)
                resp.raise_for_status()
                data = resp.json()

            items: list[SourceResult] = []
            for hit in data.get("hits") or []:
                object_id = hit.get("objectID")
                url = hit.get("url") or (
                    f"https://news.ycombinator.com/item?id={object_id}" if object_id else None
                )
                if not url:
                    continue
                points = hit.get("points")
                num_comments = hit.get("num_comments")
                engagement = None
                if isinstance(points, int) or isinstance(num_comments, int):
                    engagement = (points or 0) + (num_comments or 0)
                items.append(
                    _result(
                        title=hit.get("title") or hit.get("story_title") or "(untitled)",
                        url=url,
                        snippet=hit.get("story_text") or hit.get("comment_text") or "",
                        date=hit.get("created_at"),
                        engagement=engagement,
                        source=self.name,
                    )
                )
            items = [it for it in items if _is_recent(it.published_date, window_days)]
            return items[:limit]
        except Exception as exc:  # noqa: BLE001 — adapters never raise
            self.last_error = f"hackernews search failed: {exc}"
            log.warning("[community/hackernews] %s", exc)
            return []


# ── Bluesky — public AppView, requires a realistic UA ───────────────────────

_BSKY_SEARCH_URL = "https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts"


class BlueskySource:
    """Bluesky search via the public AppView. 403 (default-UA block or edge
    throttle) degrades to an empty list with the cause on ``last_error``."""

    name = "bluesky"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.last_error: str | None = None

    async def search(
        self, topic: str, *, window_days: int = 30, limit: int = 20
    ) -> list[SourceResult]:
        self.last_error = None
        params = {
            "q": topic,
            "sort": "top",
            "since": _since_rfc3339(window_days),
            "limit": str(min(limit, 100)),
        }
        # A realistic browser UA is required on EVERY request — default UAs get 403.
        headers = {"User-Agent": _BROWSER_UA}
        try:
            async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
                async with ratelimit.acquire("bluesky"):
                    resp = await client.get(_BSKY_SEARCH_URL, params=params, headers=headers)
                _record_429("bluesky", resp)
                resp.raise_for_status()
                data = resp.json()

            items: list[SourceResult] = []
            for post in data.get("posts") or []:
                record = post.get("record") or {}
                text = record.get("text") or ""
                author = post.get("author") or {}
                handle = author.get("handle")
                uri = post.get("uri") or ""
                rkey = uri.rsplit("/", 1)[-1] if uri else None
                url = f"https://bsky.app/profile/{handle}/post/{rkey}" if handle and rkey else uri
                if not url:
                    continue
                engagement = (post.get("likeCount") or 0) + (post.get("repostCount") or 0)
                title = text[:80] + ("…" if len(text) > 80 else "") if text else "(bluesky post)"
                items.append(
                    _result(
                        title=title,
                        url=url,
                        snippet=text,
                        date=record.get("createdAt"),
                        engagement=engagement,
                        source=self.name,
                    )
                )
            items = [it for it in items if _is_recent(it.published_date, window_days)]
            return items[:limit]
        except Exception as exc:  # noqa: BLE001 — adapters never raise
            self.last_error = f"bluesky search failed: {exc}"
            log.warning("[community/bluesky] %s", exc)
            return []


# ── GitHub — /search/repositories + /search/issues ──────────────────────────

_GH_REPOS_URL = "https://api.github.com/search/repositories"
_GH_ISSUES_URL = "https://api.github.com/search/issues"
_GH_QUERY_MAX_LEN = 200  # headroom under GitHub's 256-char query cap for qualifiers


class GitHubSource:
    """GitHub search over repositories + issues. An optional ``github_token``
    both authenticates the request (via the ``Authorization`` header) and lifts
    the rate limit from 10 to 30 req/min."""

    name = "github"

    def __init__(self, settings: Settings) -> None:
        self._token = settings.github_token
        self.last_error: str | None = None

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/vnd.github+json", "User-Agent": _API_UA}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    async def _one(
        self, client: httpx.AsyncClient, url: str, query: str, per_page: int
    ) -> list[dict]:
        params = {"q": query, "sort": "updated", "order": "desc", "per_page": str(per_page)}
        rpm = 30.0 if self._token else None  # None → module default (10)
        try:
            async with ratelimit.acquire("github_search", rpm=rpm):
                resp = await client.get(url, params=params, headers=self._headers())
            _record_429("github_search", resp)
            resp.raise_for_status()
            return resp.json().get("items") or []
        except Exception as exc:  # noqa: BLE001 — adapters never raise
            self.last_error = f"github {url} failed: {exc}"
            log.warning("[community/github] %s failed: %s", url, exc)
            return []

    async def search(
        self, topic: str, *, window_days: int = 30, limit: int = 20
    ) -> list[SourceResult]:
        self.last_error = None
        query = topic.strip()[:_GH_QUERY_MAX_LEN]
        since = _since_date_str(window_days)
        per_type = max(1, limit // 2)

        async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
            repo_items, issue_items = await asyncio.gather(
                self._one(client, _GH_REPOS_URL, f"{query} pushed:>={since}", per_type),
                self._one(client, _GH_ISSUES_URL, f"{query} updated:>={since}", per_type),
            )

        # _one already isolates fetch failures to []; this parse loop gets its own
        # try so a malformed-but-200 item degrades to [] rather than raising.
        try:
            items: list[SourceResult] = []
            for repo in repo_items:
                url = repo.get("html_url")
                if not url:
                    continue
                items.append(
                    _result(
                        title=repo.get("full_name") or repo.get("name") or "(repo)",
                        url=url,
                        snippet=repo.get("description") or "",
                        date=repo.get("pushed_at") or repo.get("updated_at"),
                        engagement=repo.get("stargazers_count"),
                        source=self.name,
                    )
                )
            for issue in issue_items:
                url = issue.get("html_url")
                if not url:
                    continue
                reactions = issue.get("reactions") or {}
                total_reactions = (
                    reactions.get("total_count") if isinstance(reactions, dict) else None
                )
                comments = issue.get("comments")
                engagement = None
                if isinstance(total_reactions, int) or isinstance(comments, int):
                    engagement = (total_reactions or 0) + (comments or 0)
                items.append(
                    _result(
                        title=issue.get("title") or "(issue)",
                        url=url,
                        snippet=(issue.get("body") or "")[:280],
                        date=issue.get("updated_at") or issue.get("created_at"),
                        engagement=engagement,
                        source=self.name,
                    )
                )
            items = [it for it in items if _is_recent(it.published_date, window_days)]
            return items[:limit]
        except Exception as exc:  # noqa: BLE001 — adapters never raise
            self.last_error = f"github parsing failed: {exc}"
            log.warning("[community/github] parsing failed: %s", exc)
            return []


# ── Reddit — OAuth-first, public-JSON fallback, ScrapeCreators last resort ───

_REDDIT_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
_REDDIT_OAUTH_SEARCH_URL = "https://oauth.reddit.com/search"
_REDDIT_PUBLIC_SEARCH_URL = "https://www.reddit.com/search.json"
_SCRAPECREATORS_REDDIT_URL = "https://api.scrapecreators.com/v1/reddit/search"


def _window_to_reddit_t(window_days: int) -> str:
    if window_days <= 1:
        return "day"
    if window_days <= 7:
        return "week"
    if window_days <= 31:
        return "month"
    if window_days <= 366:
        return "year"
    return "all"


def _parse_reddit_children(children: list[dict]) -> list[SourceResult]:
    items: list[SourceResult] = []
    for child in children:
        d = child.get("data") or {}
        permalink = d.get("permalink")
        url = f"https://reddit.com{permalink}" if permalink else d.get("url")
        if not url:
            continue
        created = d.get("created_utc")
        date = None
        if isinstance(created, (int, float)):
            date = datetime.fromtimestamp(created, tz=timezone.utc).isoformat()
        score = d.get("score")
        num_comments = d.get("num_comments")
        engagement = None
        if isinstance(score, int) or isinstance(num_comments, int):
            engagement = (score or 0) + (num_comments or 0)
        items.append(
            _result(
                title=d.get("title") or "(reddit post)",
                url=url,
                snippet=d.get("selftext") or "",
                date=date,
                engagement=engagement,
                source="reddit",
            )
        )
    return items


class RedditSource:
    """Reddit search with a three-tier fallback chain. OAuth is the default path
    when credentials are set (Reddit blocks non-browser clients at the TLS edge);
    unauthenticated public JSON is the fallback; ScrapeCreators is the last
    resort when its key is set. ``last_path`` records which tier served."""

    name = "reddit"

    def __init__(self, settings: Settings) -> None:
        self._client_id = settings.reddit_client_id
        self._client_secret = settings.reddit_client_secret
        self._scrapecreators_key = settings.scrapecreators_api_key
        self.last_error: str | None = None
        self.last_path: str | None = None

    async def _oauth_token(self) -> str | None:
        basic = base64.b64encode(
            f"{self._client_id}:{self._client_secret}".encode()
        ).decode()
        try:
            async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
                async with ratelimit.acquire("reddit"):
                    resp = await client.post(
                        _REDDIT_TOKEN_URL,
                        data={"grant_type": "client_credentials"},
                        headers={"Authorization": f"Basic {basic}", "User-Agent": _API_UA},
                    )
                _record_429("reddit", resp)
                resp.raise_for_status()
                return resp.json().get("access_token")
        except Exception as exc:  # noqa: BLE001 — adapters never raise
            self.last_error = f"reddit OAuth token request failed: {exc}"
            log.warning("[community/reddit] OAuth token request failed: %s", exc)
            return None

    async def _oauth_search(
        self, token: str, topic: str, *, window_days: int, limit: int
    ) -> list[SourceResult] | None:
        params = {
            "q": topic,
            "sort": "relevance",
            "t": _window_to_reddit_t(window_days),
            "limit": str(limit),
        }
        headers = {"Authorization": f"Bearer {token}", "User-Agent": _API_UA}
        try:
            async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
                async with ratelimit.acquire("reddit"):
                    resp = await client.get(
                        _REDDIT_OAUTH_SEARCH_URL, params=params, headers=headers
                    )
                _record_429("reddit", resp)
                resp.raise_for_status()
                data = resp.json()
            children = (data.get("data") or {}).get("children") or []
            return _parse_reddit_children(children)
        except Exception as exc:  # noqa: BLE001 — adapters never raise
            self.last_error = f"reddit OAuth search failed: {exc}"
            log.warning("[community/reddit] OAuth search failed: %s", exc)
            return None

    async def _public_search(
        self, topic: str, *, window_days: int, limit: int
    ) -> list[SourceResult] | None:
        params = {
            "q": topic,
            "sort": "relevance",
            "t": _window_to_reddit_t(window_days),
            "limit": str(limit),
        }
        headers = {"User-Agent": _BROWSER_UA}
        try:
            async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
                async with ratelimit.acquire("reddit"):
                    resp = await client.get(
                        _REDDIT_PUBLIC_SEARCH_URL, params=params, headers=headers
                    )
                _record_429("reddit", resp)
                if resp.status_code != 200:
                    self.last_error = f"reddit public search returned HTTP {resp.status_code}"
                    log.warning("[community/reddit] public search HTTP %s", resp.status_code)
                    return None
                if "json" not in resp.headers.get("content-type", ""):
                    self.last_error = "reddit public search returned non-JSON (anti-bot page)"
                    log.warning("[community/reddit] public search returned non-JSON")
                    return None
                data = resp.json()
            children = (data.get("data") or {}).get("children") or []
            return _parse_reddit_children(children)
        except Exception as exc:  # noqa: BLE001 — adapters never raise
            self.last_error = f"reddit public search failed: {exc}"
            log.warning("[community/reddit] public search failed: %s", exc)
            return None

    async def _scrapecreators_search(
        self, topic: str, *, window_days: int, limit: int
    ) -> list[SourceResult] | None:
        headers = {"x-api-key": self._scrapecreators_key or ""}
        params = {"query": topic, "limit": str(limit)}
        try:
            async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
                async with ratelimit.acquire("scrapecreators"):
                    resp = await client.get(
                        _SCRAPECREATORS_REDDIT_URL, params=params, headers=headers
                    )
                _record_429("scrapecreators", resp)
                resp.raise_for_status()
                data = resp.json()

            # ScrapeCreators' Reddit response shape isn't independently
            # doc-verified — tolerate both a raw-Reddit-like and a flatter
            # ``results[]`` shape.
            results = data.get("results") or data.get("posts") or []
            items: list[SourceResult] = []
            for r in results:
                url = r.get("url") or r.get("permalink")
                if url and url.startswith("/"):
                    url = f"https://reddit.com{url}"
                if not url:
                    continue
                created = r.get("created_utc") or r.get("created_at")
                date = None
                if isinstance(created, (int, float)):
                    date = datetime.fromtimestamp(created, tz=timezone.utc).isoformat()
                elif isinstance(created, str):
                    date = created
                score = r.get("score") or r.get("upvotes")
                num_comments = r.get("num_comments") or r.get("comments")
                engagement = None
                if isinstance(score, int) or isinstance(num_comments, int):
                    engagement = (score or 0) + (num_comments or 0)
                items.append(
                    _result(
                        title=r.get("title") or "(reddit post)",
                        url=url,
                        snippet=r.get("selftext") or r.get("text") or "",
                        date=date,
                        engagement=engagement,
                        source=self.name,
                    )
                )
            return items[:limit]
        except Exception as exc:  # noqa: BLE001 — adapters never raise
            self.last_error = f"reddit ScrapeCreators search failed: {exc}"
            log.warning("[community/reddit] ScrapeCreators search failed: %s", exc)
            return None

    async def search(
        self, topic: str, *, window_days: int = 30, limit: int = 20
    ) -> list[SourceResult]:
        self.last_error = None
        self.last_path = None

        if self._client_id and self._client_secret:
            token = await self._oauth_token()
            if token:
                result = await self._oauth_search(
                    token, topic, window_days=window_days, limit=limit
                )
                if result is not None:
                    self.last_path = "oauth"
                    return [it for it in result if _is_recent(it.published_date, window_days)][
                        :limit
                    ]

        result = await self._public_search(topic, window_days=window_days, limit=limit)
        if result is not None:
            self.last_path = "public"
            return [it for it in result if _is_recent(it.published_date, window_days)][:limit]

        if self._scrapecreators_key:
            result = await self._scrapecreators_search(
                topic, window_days=window_days, limit=limit
            )
            if result is not None:
                self.last_path = "scrapecreators"
                return [it for it in result if _is_recent(it.published_date, window_days)][
                    :limit
                ]

        if self.last_error is None:
            self.last_error = "reddit: no credentials configured and public endpoint unavailable"
        log.warning("[community/reddit] %s", self.last_error)
        return []


# ── X — ScrapeCreators-backed, handle-based ─────────────────────────────────

_SCRAPECREATORS_X_TWEETS_URL = "https://api.scrapecreators.com/v1/twitter/user-tweets"


class XSource:
    """Handle-based X search. ScrapeCreators documents ``/v1/twitter/user-tweets``
    but no keyword-search endpoint, so this pulls recent posts from a configured
    watch list (``x_handles``) and filters them against the topic client-side.
    Inactive (returns ``[]``, no network) without both a key and a watch list."""

    name = "x"

    def __init__(self, settings: Settings) -> None:
        self._key = settings.scrapecreators_api_key
        self._handles = [h.strip().lstrip("@") for h in settings.x_handles if h and h.strip()]
        self.active = bool(self._key and self._handles)
        self.last_error: str | None = None

    async def _fetch_handle(
        self,
        client: httpx.AsyncClient,
        handle: str,
        headers: dict[str, str],
        *,
        topic_terms: set[str],
        since: datetime,
        per_handle: int,
    ) -> list[SourceResult]:
        try:
            async with ratelimit.acquire("scrapecreators"):
                resp = await client.get(
                    _SCRAPECREATORS_X_TWEETS_URL, params={"handle": handle}, headers=headers
                )
            _record_429("scrapecreators", resp)
            resp.raise_for_status()
            data = resp.json()

            out: list[SourceResult] = []
            for tweet in data.get("tweets") or []:
                text = tweet.get("text") or ""
                if topic_terms and not any(term in text.lower() for term in topic_terms):
                    continue
                created_at = tweet.get("created_at")
                tweet_date = _parse_any_date(created_at) if created_at else None
                if tweet_date is not None and tweet_date < since:
                    continue
                tweet_id = tweet.get("id") or tweet.get("tweet_id")
                url = tweet.get("url") or (
                    f"https://x.com/{handle}/status/{tweet_id}" if tweet_id else None
                )
                if not url:
                    continue
                likes = tweet.get("like_count") or tweet.get("favorite_count") or 0
                reposts = tweet.get("retweet_count") or 0
                title = text[:80] + ("…" if len(text) > 80 else "") if text else "(post)"
                out.append(
                    _result(
                        title=title,
                        url=url,
                        snippet=text,
                        date=created_at,
                        engagement=likes + reposts,
                        source=self.name,
                    )
                )
                if len(out) >= per_handle:
                    break
            return out
        except Exception as exc:  # noqa: BLE001 — adapters never raise
            self.last_error = f"x user-tweets fetch failed for @{handle}: {exc}"
            log.warning("[community/x] user-tweets fetch failed for @%s: %s", handle, exc)
            return []

    async def search(
        self, topic: str, *, window_days: int = 30, limit: int = 20
    ) -> list[SourceResult]:
        self.last_error = None
        if not self.active:
            return []

        topic_terms = {t.lower() for t in topic.split() if len(t) >= 3}
        since = datetime.now(timezone.utc) - timedelta(days=window_days)
        headers = {"x-api-key": self._key or ""}
        per_handle = max(1, limit // len(self._handles))

        async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
            # return_exceptions as defense-in-depth: _fetch_handle already
            # isolates failures, but one bad handle must never sink the others.
            results = await asyncio.gather(
                *(
                    self._fetch_handle(
                        client,
                        h,
                        headers,
                        topic_terms=topic_terms,
                        since=since,
                        per_handle=per_handle,
                    )
                    for h in self._handles
                ),
                return_exceptions=True,
            )
        flattened = [item for sub in results if isinstance(sub, list) for item in sub]
        return flattened[:limit]


# ── YouTube — optional, own 100-searches/day quota bucket ────────────────────

_YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"


class YouTubeSource:
    """YouTube search via the Data API. ``search.list`` has its own
    100-searches/day quota bucket, so this issues exactly one call per run.
    Inactive (returns ``[]``, no network) without ``youtube_api_key``."""

    name = "youtube"

    def __init__(self, settings: Settings) -> None:
        self._key = settings.youtube_api_key
        self.active = bool(self._key)
        self.last_error: str | None = None

    async def search(
        self, topic: str, *, window_days: int = 30, limit: int = 20
    ) -> list[SourceResult]:
        self.last_error = None
        if not self._key:
            return []

        params = {
            "part": "snippet",
            "q": topic,
            "type": "video",
            "order": "relevance",
            "publishedAfter": _since_rfc3339(window_days),
            "maxResults": str(min(limit, 50)),
            "key": self._key,
        }
        try:
            async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
                resp = await client.get(_YOUTUBE_SEARCH_URL, params=params)
                resp.raise_for_status()
                data = resp.json()

            items: list[SourceResult] = []
            for entry in data.get("items") or []:
                video_id = (entry.get("id") or {}).get("videoId")
                if not video_id:
                    continue
                snippet_data = entry.get("snippet") or {}
                items.append(
                    _result(
                        title=snippet_data.get("title") or "(video)",
                        url=f"https://www.youtube.com/watch?v={video_id}",
                        snippet=snippet_data.get("description") or "",
                        date=snippet_data.get("publishedAt"),
                        engagement=None,  # search.list doesn't return view/like counts
                        source=self.name,
                    )
                )
            return items[:limit]
        except Exception as exc:  # noqa: BLE001 — adapters never raise
            self.last_error = f"youtube search failed: {exc}"
            log.warning("[community/youtube] %s", exc)
            return []


__all__ = [
    "RedditSource",
    "HackerNewsSource",
    "BlueskySource",
    "GitHubSource",
    "XSource",
    "YouTubeSource",
]
