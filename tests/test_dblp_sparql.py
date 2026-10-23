from types import SimpleNamespace
import json

import pytest

from zotero_arxiv_daily.dblp_sparql import DblpSparqlLookup, SCHEMA, build_query, convert_response, title_variants


def binding(**overrides):
    data = dict(p="https://dblp.org/rec/conf/nips/example", title="A published paper.", venue="NeurIPS", year="2023",
                kind=SCHEMA + "Inproceedings", ee="http://papers.nips.cc/paper/2023/hash/example-Abstract.html", author="Test Author")
    data.update(overrides)
    return {k: {"value": v} for k, v in data.items()}


def test_adapter_preserves_publication_identity_and_authors():
    payload = {"results": {"bindings": [binding(), binding(author="Other Author")]}}
    info = convert_response(payload)["result"]["hits"]["hit"][0]["info"]
    assert info["key"] == "conf/nips/example"
    assert info["venue"] == "NeurIPS" and info["type"] == "Conference and Workshop Papers"
    assert info["authors"]["author"] == ["Test Author", "Other Author"]
    assert info["ee"] == ["https://papers.nips.cc/paper/2023/hash/example-Abstract.html"]


def test_adapter_cannot_invent_journal_type_for_preprint():
    payload = {"results": {"bindings": [binding(kind=SCHEMA + "Informal")]}}
    assert convert_response(payload)["result"]["hits"]["hit"] == []


@pytest.mark.parametrize("payload", [{}, {"results": {}}, {"results": {"bindings": "invalid"}}, {"results": {"bindings": [binding()] * 2000}}])
def test_bad_or_truncated_response_fails(payload):
    with pytest.raises(ValueError):
        convert_response(payload)


def test_title_is_a_quoted_literal_not_executable_query():
    title = 'Paper" } SERVICE <https://evil.example> { #'
    query = build_query(title, 10)
    assert json.dumps(title) in query
    assert "LIMIT 10" in query and "LIMIT 2000" in query


def test_typographic_variants_keep_words():
    variants = title_variants("  End-to-End Object Detection\nwith Transformers  ")
    assert "End-to-End Object Detection with Transformers." in variants
    assert "End-to-End Object Detection with Transformers" in variants
    assert all("Detection" in t or "detection" in t for t in variants)


def test_challenge_response_never_becomes_empty_success():
    calls = []
    def get(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(status_code=202)
    client = DblpSparqlLookup(session=SimpleNamespace(get=get), sleep_fn=lambda _: None)
    with pytest.raises(LookupError):
        client.lookup("Paper")
    assert len(calls) == 2
    assert all(call["timeout"] == 15 for call in calls)


def test_successful_empty_lookup_is_distinct_from_network_failure():
    client = DblpSparqlLookup(session=SimpleNamespace(get=lambda *a, **kw: SimpleNamespace(
        status_code=200, json=lambda: {"results": {"bindings": []}})))
    assert client.lookup("Not indexed")["result"]["hits"]["hit"] == []
