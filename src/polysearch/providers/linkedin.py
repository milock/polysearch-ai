"""Opt-in LinkedIn profile enrichment via the ScrapeCreators API.

Pulls a public LinkedIn profile (``GET /v1/linkedin/profile?url=...``) and
normalizes it into a single ``SourceResult`` for PERSON-mode research. The
standard ScrapeCreators tier masks detailed employment history and education
with asterisks, so this module surfaces only the fields that come back unmasked
(about, recent posts, articles, projects, recommendations, location) and
explicitly flags the masked sections instead of emitting garbled asterisk
strings.

**Opt-in and paid.** This layer activates only when ``scrapecreators_api_key``
is set. ScrapeCreators is a paid third-party API, and the caller is responsible
for compliance with LinkedIn's Terms of Service when fetching profile data.
"""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel

from polysearch.config import Settings
from polysearch.output.schema import SourceResult

_BASE = "https://api.scrapecreators.com"
_TIMEOUT_S = 40.0


class LinkedInPost(BaseModel):
    date: str | None = None
    text: str
    url: str | None = None


class LinkedInArticle(BaseModel):
    date: str | None = None
    headline: str
    url: str | None = None


class LinkedInProject(BaseModel):
    name: str
    date_range: str | None = None
    description: str | None = None


class LinkedInProfile(BaseModel):
    url: str
    name: str | None = None
    location: str | None = None
    followers: int | None = None
    about: str | None = None
    recent_posts: list[LinkedInPost] = []
    articles: list[LinkedInArticle] = []
    projects: list[LinkedInProject] = []
    recommendations: list[str] = []
    experience_locations: list[str] = []  # unmasked location breadcrumbs only
    masked_sections: list[str] = []  # e.g. ["experience (employment history)", "education"]
    credits_remaining: int | None = None


def _masked(value: Any) -> bool:
    """True if a string value is asterisk-masked by the API tier (or empty)."""
    if not isinstance(value, str):
        return True
    stripped = value.replace(" ", "")
    if not stripped:
        return True
    stars = sum(1 for ch in stripped if ch == "*")
    return stars / len(stripped) >= 0.5


def _clean(value: Any) -> str | None:
    """Return a trimmed string, or None if missing/masked."""
    if not isinstance(value, str):
        return None
    v = value.strip()
    return None if (not v or _masked(v)) else v


def _parse_profile(url: str, d: dict[str, Any]) -> LinkedInProfile | None:
    """Normalize a raw ScrapeCreators payload into a ``LinkedInProfile``.

    Returns ``None`` when the API reports failure or returns nothing.
    """
    if not d or not d.get("success"):
        return None

    posts: list[LinkedInPost] = []
    for p in d.get("recentPosts", []) or []:
        # ScrapeCreators stores the full post body in the `title` field.
        text = _clean(p.get("title"))
        if text:
            posts.append(
                LinkedInPost(
                    date=(p.get("datePublished") or "")[:10] or None,
                    text=text,
                    url=p.get("link"),
                )
            )

    articles: list[LinkedInArticle] = []
    for a in d.get("articles", []) or []:
        head = _clean(a.get("headline"))
        if head:
            articles.append(
                LinkedInArticle(
                    date=(a.get("datePublished") or "")[:10] or None,
                    headline=head,
                    url=a.get("url"),
                )
            )

    projects: list[LinkedInProject] = []
    seen_proj: set[str] = set()
    for p in d.get("projects", []) or []:
        name = _clean(p.get("name"))
        if name and name not in seen_proj:
            seen_proj.add(name)
            projects.append(
                LinkedInProject(
                    name=name,
                    date_range=p.get("dateRange"),
                    description=_clean(p.get("description")),
                )
            )

    recs: list[str] = []
    for r in d.get("recommendations", []) or []:
        txt = _clean(r.get("text"))
        if txt:
            giver = _clean(r.get("name"))
            recs.append(f"{giver}: {txt}" if giver else txt)

    exp_locs: list[str] = []
    masked_experience = False
    for e in d.get("experience", []) or []:
        if _masked(e.get("name")):
            masked_experience = True
        loc = _clean(e.get("location"))
        if loc and loc not in exp_locs:
            exp_locs.append(loc)

    masked_sections: list[str] = []
    if masked_experience:
        masked_sections.append("experience (employment history)")
    edu = d.get("education", []) or []
    if edu and all(_masked(e.get("name")) for e in edu):
        masked_sections.append("education")

    followers = d.get("followers")
    credits = d.get("credits_remaining")
    return LinkedInProfile(
        url=url,
        name=_clean(d.get("name")),
        location=_clean(d.get("location")),
        followers=followers if isinstance(followers, int) else None,
        about=_clean(d.get("about")),
        recent_posts=posts,
        articles=articles,
        projects=projects,
        recommendations=recs,
        experience_locations=exp_locs,
        masked_sections=masked_sections,
        credits_remaining=credits if isinstance(credits, int) else None,
    )


def _build_snippet(p: LinkedInProfile) -> str:
    """Compose a single-string summary of the unmasked, self-authored signal.

    Masked employment/education are named as a flag, never printed as asterisks.
    """
    parts: list[str] = []
    if p.location:
        parts.append(p.location)
    if p.about:
        parts.append(p.about)
    if p.experience_locations:
        parts.append(
            "Experience locations (employer names masked): "
            + ", ".join(p.experience_locations)
        )
    if p.recent_posts:
        first = " ".join(p.recent_posts[0].text.split())
        parts.append(f"Recent post: {first}")
    if p.masked_sections:
        parts.append("Masked by API tier: " + ", ".join(p.masked_sections))
    return " | ".join(parts) if parts else p.url


def _to_source_result(p: LinkedInProfile) -> SourceResult:
    return SourceResult(
        url=p.url,
        title=p.name or "LinkedIn profile",
        snippet=_build_snippet(p),
        tier="primary",
        published_date=None,
        layer="linkedin",
        engagement=p.followers,
    )


class LinkedInEnricher:
    """Opt-in ScrapeCreators LinkedIn enrichment.

    Active only when ``scrapecreators_api_key`` is set. When inactive, ``enrich``
    short-circuits to ``None`` without any network call and ``reason`` names the
    missing credential. On an API/transport error or an empty response, ``enrich``
    returns ``None`` and surfaces the cause on ``reason`` (mirroring the
    reason-carrying null providers in ``base``).
    """

    name = "linkedin"

    def __init__(self, settings: Settings) -> None:
        self._key = settings.scrapecreators_api_key
        self.active = bool(self._key)
        self.reason: str | None = (
            None if self.active else "SCRAPECREATORS_API_KEY not set"
        )

    async def enrich(self, person_or_company: str) -> SourceResult | None:
        """Fetch + normalize a LinkedIn profile URL into a ``SourceResult``.

        ``person_or_company`` is a public LinkedIn profile (or company) URL.
        """
        key = self._key
        if not key:
            return None
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
                resp = await client.get(
                    f"{_BASE}/v1/linkedin/profile",
                    params={"url": person_or_company},
                    headers={"x-api-key": key, "Content-Type": "application/json"},
                )
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            self.reason = f"linkedin enrichment failed: {exc}"
            return None

        profile = _parse_profile(person_or_company, data)
        if profile is None:
            self.reason = "linkedin returned no profile"
            return None
        return _to_source_result(profile)
