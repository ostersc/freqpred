"""Unit tests for freqpred/ingestion/fetchers/tv_chyron.py.

All HTTP calls are mocked — no real network calls.
"""
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from freqpred.ingestion.fetchers.tv_chyron import (
    ChyronRow,
    fetch_all,
    filter_chyrons,
    parse_and_groups,
)

_NOW = datetime(2026, 3, 23, 23, 0, 0, tzinfo=UTC)


def _make_row(
    dt: datetime = _NOW,
    channel: str = "FOXNEWSW",
    duration_s: int = 17,
    identifier_path: str = "FOXNEWSW_20260323_230100_Fox_News_Tonight/start/60",
    text: str = "TRUMP: IRAN WANTS TO MAKE A DEAL",
) -> ChyronRow:
    return ChyronRow(
        dt=dt,
        channel=channel,
        duration_s=duration_s,
        identifier_path=identifier_path,
        text=text,
    )


_SAMPLE_TSV = (
    "date_time_(UTC)\tchannel\tduration\thttps://archive.org/details/\ttext\n"
    "2026-03-23 23:01:00\tFOXNEWSW\t17\tFOXNEWSW_20260323_230100_Fox_News_Tonight/start/60\t"
    "TRUMP: IRAN WANTS TO MAKE A DEAL\n"
    "2026-03-23 23:02:00\tCNNW\t15\tCNNW_20260323_230200_CNN_Tonight/start/120\tFED CUTS RATES\n"
)


def _make_client_mock(text: str = _SAMPLE_TSV, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    if status_code >= 400:
        import httpx
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status_code}",
            request=MagicMock(),
            response=MagicMock(status_code=status_code),
        )
    else:
        resp.raise_for_status = MagicMock()
    resp.text = text
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.get = AsyncMock(return_value=resp)
    return client


# ---------------------------------------------------------------------------
# parse_and_groups
# ---------------------------------------------------------------------------


def test_parse_and_groups_simple():
    assert parse_and_groups("trump") == [["trump"]]


def test_parse_and_groups_and_or():
    result = parse_and_groups('trump AND ("communist" OR "communism")')
    assert result == [["trump"], ["communist", "communism"]]


def test_parse_and_groups_quoted_phrase():
    result = parse_and_groups('"federal reserve" AND ("rate cut" OR "interest rates")')
    assert result == [["federal reserve"], ["rate cut", "interest rates"]]


def test_parse_and_groups_null_or_empty():
    assert parse_and_groups(None) == []
    assert parse_and_groups("") == []


# ---------------------------------------------------------------------------
# filter_chyrons
# ---------------------------------------------------------------------------


def test_filter_chyrons_and_match():
    rows = [_make_row(text="TRUMP communist agenda")]
    docs = filter_chyrons(rows, [["trump"], ["communist"]])
    assert len(docs) == 1


def test_filter_chyrons_and_no_match():
    rows = [_make_row(text="TRUMP wins election")]
    docs = filter_chyrons(rows, [["trump"], ["communist"]])
    assert len(docs) == 0


def test_filter_chyrons_since_cursor():
    old_dt = datetime(2026, 3, 23, 22, 0, 0, tzinfo=UTC)
    new_dt = datetime(2026, 3, 23, 23, 0, 0, tzinfo=UTC)
    rows = [
        _make_row(dt=old_dt, text="trump communist"),
        _make_row(dt=new_dt, text="trump communist"),
    ]
    docs = filter_chyrons(rows, [["trump"]], since=old_dt)
    assert len(docs) == 1
    assert docs[0].published_at == new_dt


def test_filter_chyrons_case_insensitive():
    rows = [_make_row(text="FEDERAL RESERVE CUTS RATES")]
    docs = filter_chyrons(rows, [["federal reserve"]])
    assert len(docs) == 1


def test_filter_chyrons_raw_document_fields():
    row = _make_row(
        dt=_NOW,
        channel="CNNW",
        identifier_path="CNNW_20260323_230100_Anderson_Cooper/start/60",
        text="FEDERAL RESERVE CUTS RATES",
    )
    docs = filter_chyrons([row], [["federal reserve"]])
    assert len(docs) == 1
    doc = docs[0]
    assert doc.source_type == "tv_chyron"
    assert doc.source_name == "TVThirdEye"
    assert doc.source_url == "https://archive.org/details/CNNW_20260323_230100_Anderson_Cooper/start/60"
    assert doc.title == "CNNW: Anderson Cooper"
    assert doc.body == "FEDERAL RESERVE CUTS RATES"
    assert doc.tags == ["CNNW"]
    assert doc.published_at == _NOW


# ---------------------------------------------------------------------------
# fetch_all
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_all_parses_tsv():
    client = _make_client_mock()
    with patch("freqpred.ingestion.fetchers.tv_chyron.httpx.AsyncClient", return_value=client):
        rows = await fetch_all(lookback_hours=1.5)

    assert len(rows) == 2
    assert rows[0].channel == "FOXNEWSW"
    assert rows[0].duration_s == 17
    assert rows[0].identifier_path == "FOXNEWSW_20260323_230100_Fox_News_Tonight/start/60"
    assert rows[0].text == "TRUMP: IRAN WANTS TO MAKE A DEAL"
    assert rows[0].dt == datetime(2026, 3, 23, 23, 1, 0, tzinfo=UTC)
    assert rows[1].channel == "CNNW"


@pytest.mark.asyncio
async def test_fetch_all_timeout():
    import httpx
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.get = AsyncMock(side_effect=httpx.TimeoutException("timed out"))

    with patch("freqpred.ingestion.fetchers.tv_chyron.httpx.AsyncClient", return_value=client):
        rows = await fetch_all()

    assert rows == []


@pytest.mark.asyncio
async def test_fetch_all_http_error():
    client = _make_client_mock(status_code=503)
    with patch("freqpred.ingestion.fetchers.tv_chyron.httpx.AsyncClient", return_value=client):
        rows = await fetch_all()

    assert rows == []
