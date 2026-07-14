"""Unit tests for the native community adapters.

Each adapter implements the ``CommunitySource`` protocol (``name`` +
``async search(topic, *, window_days, limit) -> list[SourceResult]``), reads its
credentials from ``Settings`` (never the process environment), and is
failure-isolated: any internal error degrades to an empty list with a surfaced
``last_error`` — adapters never raise.

Response fixtures below mirror the real API shapes documented in the task brief
(HN Algolia ``hits[]``, Bluesky AppView ``posts[]``, GitHub ``items[]``, Reddit
listing ``data.children[]``, ScrapeCreators ``tweets[]``, YouTube ``items[]``).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx
import respx

from polysearch.community.adapters import (
    BlueskySource,
    GitHubSource,
    HackerNewsSource,
    RedditSource,
    XSource,
    YouTubeSource,
)
from polysearch.config import Settings
from polysearch.output.schema import SourceResult
from polysearch.providers.base import CommunitySource


# ── shared date helpers: keep fixtures inside the recency window ─────────────

def _recent_iso(days_ago: int = 2) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _recent_unix(days_ago: int = 2) -> int:
    return int((datetime.now(timezone.utc) - timedelta(days=days_ago)).timestamp())


def _old_iso(days_ago: int = 400) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


# ── endpoints ───────────────────────────────────────────────────────────────

_HN = "https://hn.algolia.com/api/v1/search_by_date"
_BSKY = "https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts"
_GH_REPOS = "https://api.github.com/search/repositories"
_GH_ISSUES = "https://api.github.com/search/issues"
_REDDIT_TOKEN = "https://www.reddit.com/api/v1/access_token"
_REDDIT_OAUTH = "https://oauth.reddit.com/search"
_REDDIT_PUBLIC = "https://www.reddit.com/search.json"
_SC_REDDIT = "https://api.scrapecreators.com/v1/reddit/search"
_SC_TWEETS = "https://api.scrapecreators.com/v1/twitter/user-tweets"
_YT = "https://www.googleapis.com/youtube/v3/search"


# ── Hacker News ─────────────────────────────────────────────────────────────

@respx.mock
async def test_hackernews_parses_algolia_hits() -> None:
    respx.get(_HN).mock(
        return_value=httpx.Response(
            200,
            json={
                "hits": [
                    {
                        "objectID": "40000001",
                        "title": "Show HN: a widget scheduler",
                        "url": "https://example.test/widget",
                        "points": 128,
                        "num_comments": 47,
                        "created_at": _recent_iso(),
                        "story_text": "we built a scheduler",
                    }
                ]
            },
        )
    )
    results = await HackerNewsSource(Settings()).search("widget scheduler")
    assert len(results) == 1
    r = results[0]
    assert isinstance(r, SourceResult)
    assert r.tier == "COMMUNITY"
    assert r.layer == "hackernews"
    assert r.url == "https://example.test/widget"
    assert r.engagement == 128 + 47


@respx.mock
async def test_hackernews_falls_back_to_item_url_when_no_link() -> None:
    respx.get(_HN).mock(
        return_value=httpx.Response(
            200,
            json={
                "hits": [
                    {
                        "objectID": "40000002",
                        "title": "Ask HN: how do widgets work",
                        "url": None,
                        "points": 10,
                        "num_comments": 2,
                        "created_at": _recent_iso(),
                    }
                ]
            },
        )
    )
    results = await HackerNewsSource(Settings()).search("widgets")
    assert results[0].url == "https://news.ycombinator.com/item?id=40000002"


@respx.mock
async def test_hackernews_drops_items_outside_window() -> None:
    respx.get(_HN).mock(
        return_value=httpx.Response(
            200,
            json={
                "hits": [
                    {
                        "objectID": "1",
                        "title": "recent",
                        "url": "https://example.test/a",
                        "points": 5,
                        "created_at": _recent_iso(2),
                    },
                    {
                        "objectID": "2",
                        "title": "stale",
                        "url": "https://example.test/b",
                        "points": 5,
                        "created_at": _old_iso(),
                    },
                ]
            },
        )
    )
    results = await HackerNewsSource(Settings()).search("x", window_days=30)
    assert [r.url for r in results] == ["https://example.test/a"]


@respx.mock
async def test_hackernews_degrades_to_empty_on_http_error() -> None:
    respx.get(_HN).mock(return_value=httpx.Response(500))
    source = HackerNewsSource(Settings())
    results = await source.search("x")
    assert results == []
    assert source.last_error is not None


# ── Bluesky ─────────────────────────────────────────────────────────────────

@respx.mock
async def test_bluesky_parses_posts_and_builds_permalink() -> None:
    respx.get(_BSKY).mock(
        return_value=httpx.Response(
            200,
            json={
                "posts": [
                    {
                        "uri": "at://did:plc:abc123/app.bsky.feed.post/3krxyz",
                        "author": {"handle": "alice.bsky.social"},
                        "record": {
                            "text": "widgets are great",
                            "createdAt": _recent_iso(),
                        },
                        "likeCount": 12,
                        "repostCount": 4,
                    }
                ]
            },
        )
    )
    results = await BlueskySource(Settings()).search("widgets")
    assert len(results) == 1
    r = results[0]
    assert r.layer == "bluesky"
    assert r.tier == "COMMUNITY"
    assert r.url == "https://bsky.app/profile/alice.bsky.social/post/3krxyz"
    assert r.engagement == 16


@respx.mock
async def test_bluesky_sends_realistic_user_agent() -> None:
    route = respx.get(_BSKY).mock(
        return_value=httpx.Response(200, json={"posts": []})
    )
    await BlueskySource(Settings()).search("x")
    ua = route.calls.last.request.headers.get("user-agent", "")
    assert "Mozilla/5.0" in ua


@respx.mock
async def test_bluesky_403_degrades_without_raising() -> None:
    respx.get(_BSKY).mock(return_value=httpx.Response(403, text="forbidden"))
    source = BlueskySource(Settings())
    results = await source.search("x")
    assert results == []
    assert source.last_error is not None
    assert "403" in source.last_error


# ── GitHub ──────────────────────────────────────────────────────────────────

@respx.mock
async def test_github_parses_repos_and_issues() -> None:
    respx.get(_GH_REPOS).mock(
        return_value=httpx.Response(
            200,
            json={
                "items": [
                    {
                        "full_name": "acme/widget",
                        "html_url": "https://github.com/acme/widget",
                        "description": "a widget lib",
                        "pushed_at": _recent_iso(),
                        "stargazers_count": 900,
                    }
                ]
            },
        )
    )
    respx.get(_GH_ISSUES).mock(
        return_value=httpx.Response(
            200,
            json={
                "items": [
                    {
                        "title": "widget crashes",
                        "html_url": "https://github.com/acme/widget/issues/7",
                        "body": "it broke",
                        "updated_at": _recent_iso(),
                        "reactions": {"total_count": 8},
                        "comments": 3,
                    }
                ]
            },
        )
    )
    results = await GitHubSource(Settings()).search("widget")
    layers = {r.layer for r in results}
    assert layers == {"github"}
    by_url = {r.url: r for r in results}
    assert by_url["https://github.com/acme/widget"].engagement == 900
    assert by_url["https://github.com/acme/widget/issues/7"].engagement == 11


@respx.mock
async def test_github_sends_auth_header_when_token_set() -> None:
    repos = respx.get(_GH_REPOS).mock(
        return_value=httpx.Response(200, json={"items": []})
    )
    respx.get(_GH_ISSUES).mock(return_value=httpx.Response(200, json={"items": []}))
    await GitHubSource(Settings(github_token="ghp_secret")).search("x")
    assert repos.calls.last.request.headers["authorization"] == "Bearer ghp_secret"


@respx.mock
async def test_github_no_auth_header_without_token() -> None:
    repos = respx.get(_GH_REPOS).mock(
        return_value=httpx.Response(200, json={"items": []})
    )
    respx.get(_GH_ISSUES).mock(return_value=httpx.Response(200, json={"items": []}))
    await GitHubSource(Settings()).search("x")
    assert "authorization" not in repos.calls.last.request.headers


# ── Reddit: OAuth-first → public → ScrapeCreators ───────────────────────────

def _reddit_listing(title: str, url_path: str) -> dict:
    return {
        "data": {
            "children": [
                {
                    "data": {
                        "title": title,
                        "permalink": url_path,
                        "created_utc": _recent_unix(),
                        "score": 55,
                        "num_comments": 9,
                        "selftext": "body",
                    }
                }
            ]
        }
    }


@respx.mock
async def test_reddit_oauth_path_served_when_credentials_present() -> None:
    token = respx.post(_REDDIT_TOKEN).mock(
        return_value=httpx.Response(200, json={"access_token": "tok-123"})
    )
    oauth = respx.get(_REDDIT_OAUTH).mock(
        return_value=httpx.Response(
            200, json=_reddit_listing("oauth hit", "/r/widgets/comments/1/x/")
        )
    )
    public = respx.get(_REDDIT_PUBLIC).mock(
        return_value=httpx.Response(200, json=_reddit_listing("public", "/r/x/y/"))
    )
    source = RedditSource(
        Settings(reddit_client_id="cid", reddit_client_secret="csec")
    )
    results = await source.search("widgets")
    assert token.called and oauth.called
    assert not public.called
    assert source.last_path == "oauth"
    assert results[0].layer == "reddit"
    assert results[0].url == "https://reddit.com/r/widgets/comments/1/x/"
    assert results[0].engagement == 64


@respx.mock
async def test_reddit_falls_back_to_public_json_on_oauth_failure() -> None:
    respx.post(_REDDIT_TOKEN).mock(return_value=httpx.Response(200, json={"access_token": "t"}))
    respx.get(_REDDIT_OAUTH).mock(return_value=httpx.Response(500))
    public = respx.get(_REDDIT_PUBLIC).mock(
        return_value=httpx.Response(
            200,
            json=_reddit_listing("public hit", "/r/widgets/comments/2/z/"),
            headers={"content-type": "application/json"},
        )
    )
    source = RedditSource(
        Settings(reddit_client_id="cid", reddit_client_secret="csec")
    )
    results = await source.search("widgets")
    assert public.called
    assert source.last_path == "public"
    assert results[0].url == "https://reddit.com/r/widgets/comments/2/z/"


@respx.mock
async def test_reddit_falls_back_to_scrapecreators_when_public_blocked() -> None:
    # No OAuth creds; public returns a 403 anti-bot wall; ScrapeCreators key set.
    respx.get(_REDDIT_PUBLIC).mock(return_value=httpx.Response(403, text="blocked"))
    sc = respx.get(_SC_REDDIT).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "sc hit",
                        "url": "/r/widgets/comments/3/q/",
                        "created_utc": _recent_unix(),
                        "score": 20,
                        "num_comments": 5,
                        "selftext": "b",
                    }
                ]
            },
        )
    )
    source = RedditSource(Settings(scrapecreators_api_key="sc-key"))
    results = await source.search("widgets")
    assert sc.called
    assert source.last_path == "scrapecreators"
    assert results[0].url == "https://reddit.com/r/widgets/comments/3/q/"


@respx.mock
async def test_reddit_returns_empty_when_all_paths_unavailable() -> None:
    respx.get(_REDDIT_PUBLIC).mock(return_value=httpx.Response(403, text="blocked"))
    source = RedditSource(Settings())  # no creds, no scrapecreators key
    results = await source.search("widgets")
    assert results == []
    assert source.last_error is not None


# ── X: handle-based, credential + watch-list gated ──────────────────────────

@respx.mock
async def test_x_inactive_without_key_or_handles() -> None:
    # key but no handles
    assert await XSource(Settings(scrapecreators_api_key="k")).search("widget") == []
    # handles but no key
    assert await XSource(Settings(x_handles=["acme"])).search("widget") == []


@respx.mock
async def test_x_pulls_handle_tweets_and_filters_to_topic() -> None:
    respx.get(_SC_TWEETS).mock(
        return_value=httpx.Response(
            200,
            json={
                "tweets": [
                    {
                        "id": "111",
                        "text": "our new widget just shipped",
                        "created_at": _recent_iso(),
                        "like_count": 30,
                        "retweet_count": 6,
                    },
                    {
                        "id": "222",
                        "text": "unrelated lunch photo",
                        "created_at": _recent_iso(),
                        "like_count": 100,
                        "retweet_count": 2,
                    },
                ]
            },
        )
    )
    source = XSource(
        Settings(scrapecreators_api_key="k", x_handles=["acmecorp"])
    )
    results = await source.search("widget")
    assert len(results) == 1
    r = results[0]
    assert r.layer == "x"
    assert r.url == "https://x.com/acmecorp/status/111"
    assert r.engagement == 36


@respx.mock
async def test_x_sends_api_key_header() -> None:
    route = respx.get(_SC_TWEETS).mock(
        return_value=httpx.Response(200, json={"tweets": []})
    )
    await XSource(Settings(scrapecreators_api_key="k", x_handles=["acme"])).search("widget")
    assert route.calls.last.request.headers["x-api-key"] == "k"


# ── YouTube: key-gated, single call ─────────────────────────────────────────

@respx.mock
async def test_youtube_inactive_without_key() -> None:
    assert await YouTubeSource(Settings()).search("widget") == []


@respx.mock
async def test_youtube_parses_search_results() -> None:
    route = respx.get(_YT).mock(
        return_value=httpx.Response(
            200,
            json={
                "items": [
                    {
                        "id": {"videoId": "vid123"},
                        "snippet": {
                            "title": "Widget review",
                            "description": "all about widgets",
                            "publishedAt": _recent_iso(),
                        },
                    }
                ]
            },
        )
    )
    results = await YouTubeSource(Settings(youtube_api_key="yt-key")).search("widget")
    assert len(results) == 1
    r = results[0]
    assert r.layer == "youtube"
    assert r.url == "https://www.youtube.com/watch?v=vid123"
    assert r.engagement is None
    # One call per run (own 100/day quota bucket).
    assert route.call_count == 1


# ── protocol conformance ────────────────────────────────────────────────────

def test_all_adapters_satisfy_community_source_protocol() -> None:
    s = Settings()
    for cls in (
        RedditSource,
        HackerNewsSource,
        BlueskySource,
        GitHubSource,
        XSource,
        YouTubeSource,
    ):
        assert isinstance(cls(s), CommunitySource)
