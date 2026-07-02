# Internet Archive TV News Archive — Research Notes

> **Status: implemented.** The fetcher lives at `freqpred/ingestion/fetchers/tv_archive.py` and is wired into the main ingestion scheduler. The query-construction question below was resolved with dual-format catalyst generation — the catalyst generator (Haiku) emits both a natural-language `query_text` and a Solr/Lucene `tv_query` per catalyst (see SPEC.md §7, `CatalystQuery.tv_query`). This document remains as the API reference and design rationale.

## What it is

The [Internet Archive TV News Archive](https://archive.org/details/tv) is a searchable corpus of ~3 million U.S. television broadcasts with closed-caption transcripts. Data runs from July 2009 to **present day** (verified current as of March 20, 2026).

- 163+ stations: CNN, MSNBC, Fox News, BBC News, CSPAN, Al Jazeera, Univision, and many local affiliates
- Full closed-caption transcripts indexed and searchable
- No authentication required, no API key

**Important:** The GDELT 2.0 Television API (`api.gdeltproject.org/api/v2/tv/tv`) is a *separate, stale* product built on an older snapshot of this data — its results are years out of date. Do not use it. Use the archive.org search API directly.

---

## API Endpoint

The website's own search backend, reverse-engineered from browser network traffic:

```
GET https://archive.org/services/search/beta/page_production/
```

### Parameters

| Parameter | Description | Example |
|---|---|---|
| `service_backend` | Must be `tvs` for TV search | `tvs` |
| `user_query` | Search query (supports boolean syntax) | `trump AND ("communist" OR "communism")` |
| `hits_per_page` | Results per page | `20` |
| `page` | Page number (1-indexed) | `1` |
| `sort` | Sort order. Omit for relevance (`_score`); use `publicdate:desc` for recency | `publicdate:desc` |
| `aggregations` | Set to `false` to skip facet counts | `false` |
| `filter_map` | JSON-encoded dict of filters (see below) | see below |

### filter_map

`filter_map` is a JSON object where keys are field names and values are dicts of `{value: operator}`. Operators: `"inc"` (include), `"exc"` (exclude), `"gte"` (≥), `"lte"` (≤).

```json
{
  "date":     {"2026-03-16": "gte", "2026-03-20": "lte"},
  "year":     {"2026": "inc"},
  "language": {"English": "inc"},
  "program":  {"News": "inc"},
  "creator":  {"FOXNEWSW": "exc"}
}
```

**Date format:** `YYYY-MM` (year-month) or `YYYY-MM-DD` for day-level. The `gte`/`lte` operators on `date` create a range. For freqpred, using the market's window (e.g. from market open to close time) is the right approach.

**Sort strategy:**
- Use **relevance** (omit `sort`) when you want the most topically relevant clips — good for general catalyst queries
- Use **`publicdate:desc`** when you want the freshest clips — good for monitoring recent statements

### Query syntax

The search engine supports standard Lucene/Solr boolean syntax:

| Operator | Effect | Example |
|---|---|---|
| `AND` | Narrows — both terms must appear | `trump AND iran` |
| `OR` | Widens — either term | `"communist" OR "communism"` |
| `AND NOT` | Excludes | `trump AND NOT biden` |
| `( )` | Grouping | `trump AND ("communist" OR "communism")` |
| `" "` | Exact phrase match | `"communist china"` |
| `[ ]` | Range queries (where supported) | — |

**Key for freqpred:** For "Will X say Y?" markets, use `AND` + quotes to require both the speaker and the exact word appear in the same transcript clip:
```
trump AND ("communist" OR "communism")
```

---

## Response structure

Each hit is a `tv_clip` object. Relevant fields:

```json
{
  "fields": {
    "identifier": "KNTV_20260320_140000_Today",
    "title": "Today : KNTV : March 20, 2026 7:00am-9:00am PDT",
    "date": "2026-03-20T00:00:00Z",
    "creator": ["KNTV"],
    "collection": ["TV-KNTV", "tvarchive", "tvnews"],
    "subject": ["iran", "trump", "israel", ...],
    "start": "885",
    "__href__": "/details/KNTV_20260320_140000_Today/start/885/end/945?q=trump+iran",
    "__img__": "/download/.../thumbnail.jpg",
    "cc_excerpt": "raw caption text from start of the clip block (not query-relevant)"
  },
  "highlight": {
    "text": ["what's motivating president {{{trump}}} to continue with this war in {{{iran}}}?..."]
  }
}
```

### Fields to use for Document storage

| Field | Maps to | Notes |
|---|---|---|
| `highlight.text[0]` | `Document.content` | The relevant transcript excerpt. Strip `{{{ }}}` markers. This is what we embed. |
| `https://archive.org` + `__href__` | `Document.source_url` | Unique, deep-links to exact timestamp in broadcast |
| `fields.title` | `Document.title` | Station + show + date/time |
| `fields.date` | `Document.published_at` | Broadcast date |
| `fields.subject` | metadata | Auto-extracted named entities (free NER) — people, places, orgs |
| `fields.start` | metadata | Timestamp offset in seconds within the broadcast |
| `cc_excerpt` | discard | Raw caption from block start, not query-relevant |

The `highlight.text` field is the gold — it contains the transcript excerpt that matched the query, with matched terms wrapped in `{{{ }}}` for highlighting. After stripping markers this is the document body we embed and store.

---

## Query construction for freqpred

### The architecture problem (resolved)

The ingestion scheduler originally passed a plain `query_text: str` to each fetcher — no market context. Two options were considered:

- **Option A — per-fetcher query translation:** catalyst generator emits plain keywords; the fetcher mechanically joins tokens with `AND`. No schema changes, but the transform can't know which terms must be *said on air* vs merely topical.
- **Option B — catalyst generator emits TV-aware queries:** the catalyst prompt produces proper boolean syntax alongside the web query.

**Option B won.** `CatalystQuery` carries both `query_text` (natural-language web search for Tavily/NewsAPI/GDELT/Reddit) and `tv_query` (Solr/Lucene boolean syntax), generated in one Haiku call; `tv_query` is `null` when TV transcripts aren't a useful signal for that catalyst. The scheduler passes `tv_query` to `tv_archive.fetch()` (and reuses it for Guardian, which also accepts Solr syntax); `tv_chyron.py` reuses the same `tv_query` AND-groups for local chyron filtering. Stored documents use `source_type="tv_transcript"`.

### Catalyst design for TV

The TV archive indexes *what people said*, not what was written about them. This changes how catalysts are framed (and is reflected in the catalyst generator prompt):

- **Good:** `trump AND ("communist" OR "communism")` — finds Trump saying the word
- **Bad:** `trump communist policy china` — too broad, finds any clip mentioning Trump near communist
- **Good:** `trump AND "tariffs" AND china` — finds clips where Trump discusses tariffs with China mentioned
- **Bad:** `"will trump impose tariffs"` — no one says this phrase on air

For "Will X say Y?" markets specifically, the pattern is clear and mechanical. For more complex probability markets ("Will the Fed cut rates in Q2?"), the catalysts need to surface *evidence* — expert commentary, Fed chair statements, economic data being reported. The query `"federal reserve" AND ("rate cut" OR "interest rates")` gets closer.

---

## Rate limits / reliability

- No documented rate limits
- No authentication required
- Response time: ~30s on cache miss, fast on cache hit (TTL ~20 min based on `Cache-Control` headers)
- The API is an internal endpoint reverse-engineered from the website — not a published API. Could change without notice.

---

## Implementation summary

1. ✅ Query construction: Option B — dual-format catalyst generation (`query_text` + `tv_query`)
2. ✅ Fetcher: `freqpred/ingestion/fetchers/tv_archive.py`, same pattern as the GDELT fetcher
3. ✅ Wired into the main ingestion scheduler's per-query loop (fires only when `tv_query` is present)
4. ✅ Documents stored with `source_type="tv_transcript"`; the related Third Eye chyron fetcher (`tv_chyron.py`, `source_type="tv_chyron"`) runs in the realtime scheduler — see SPEC.md §9
