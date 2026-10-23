"""Safety boundaries between retrieval, ranking, and email delivery."""
from email import message_from_string
from types import SimpleNamespace

import feedparser
import pytest
from omegaconf import open_dict

from tests.canned_responses import make_sample_paper, make_stub_openai_client, make_stub_smtp, make_stub_zotero_client
from zotero_arxiv_daily.construct_email import render_email
from zotero_arxiv_daily.executor import Executor
from zotero_arxiv_daily.retriever import arxiv_retriever as module


EVIDENCE = dict(venue="CVPR", year=2025, publication_url="https://openaccess.thecvf.com/paper.html",
                verification_url="https://dblp.org/rec/conf/cvpr/example", topics=["visual_detection"], evidence="dblp")


def strict(config):
    with open_dict(config):
        config.curation.enabled = True
    return config


def stub_dependencies(monkeypatch, papers):
    import smtplib
    client = make_stub_openai_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: make_stub_zotero_client())
    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", lambda **kw: client)
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI", lambda **kw: client)
    monkeypatch.setattr(module.ArxivRetriever, "retrieve_papers", lambda self: papers)
    sent = []
    monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp(sent))
    return sent


def test_rejected_candidates_never_download_fulltext(config, monkeypatch):
    entries = [feedparser.FeedParserDict(id=f"oai:arXiv.org:2601.0000{i}", title=t,
              author="Test Author", summary="test", link=f"https://arxiv.org/abs/2601.0000{i}")
              for i, t in enumerate(["Unqualified paper", "Object detection method"])]
    monkeypatch.setattr(module.feedparser, "parse", lambda url: SimpleNamespace(
        status=200, bozo=False, feed=SimpleNamespace(title="arxiv feed"), entries=entries))
    class Gate:
        enabled = True
        report = {"accepted": 1, "rejected": 1}
        def qualify(self, p):
            return dict(EVIDENCE) if p.title == "Object detection method" else None
    monkeypatch.setattr(module, "CurationGate", lambda cfg: Gate())
    downloads = []
    monkeypatch.setattr(module, "extract_text_from_tar", lambda p: downloads.append(p.title) or "Full text")
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)
    papers = module.ArxivRetriever(strict(config)).retrieve_papers()
    assert downloads == ["Object detection method"]
    assert [p.title for p in papers] == downloads
    assert papers[0].publication == EVIDENCE


def test_debug_never_skips_curation(config, monkeypatch):
    config.executor.debug = True
    entries = [feedparser.FeedParserDict(id=f"oai:arXiv.org:2601.{i:05}", title=f"Paper {i}",
               author="Author", summary="test") for i in range(12)]
    monkeypatch.setattr(module.feedparser, "parse", lambda url: SimpleNamespace(
        status=200, bozo=False, feed=SimpleNamespace(title="arxiv feed"), entries=entries))
    seen = []
    class Gate:
        enabled = True
        report = {}
        def qualify(self, p):
            seen.append(p.title)
            return EVIDENCE if p.title == "Paper 11" else None
    monkeypatch.setattr(module, "CurationGate", lambda cfg: Gate())
    papers = module.ArxivRetriever(strict(config))._retrieve_raw_papers()
    assert len(seen) == 12
    assert [p.title for p in papers] == ["Paper 11"]


def test_verification_outage_is_not_reported_as_no_matching_papers(config, monkeypatch):
    monkeypatch.setattr(module.feedparser, "parse", lambda url: SimpleNamespace(
        status=200, bozo=False, feed=SimpleNamespace(title="arxiv feed"), entries=[]))
    gate = SimpleNamespace(enabled=True, report={"lookup_failed": 1})
    monkeypatch.setattr(module, "CurationGate", lambda cfg: gate)
    with pytest.raises(RuntimeError, match="verification was incomplete"):
        module.ArxivRetriever(strict(config))._retrieve_raw_papers()


def test_conversion_without_evidence_cannot_download(config, monkeypatch):
    retriever = module.ArxivRetriever(strict(config))
    with pytest.raises(ValueError, match="verified publication evidence"):
        retriever.convert_to_paper(SimpleNamespace(title="Paper"))


def test_unqualified_paper_cannot_reach_email(config, monkeypatch):
    sent = stub_dependencies(monkeypatch, [make_sample_paper()])
    with pytest.raises(ValueError, match="without verified"):
        Executor(strict(config)).run()
    assert sent == []


def test_other_sources_cannot_bypass_gate(config):
    strict(config)
    config.executor.source = ["arxiv", "biorxiv"]
    with pytest.raises(ValueError, match="only source"):
        Executor(config)


def test_daily_workflow_rejects_accidentally_disabled_gate(config, monkeypatch):
    monkeypatch.setenv("REQUIRE_STRICT_CURATION", "true")
    with pytest.raises(ValueError, match="requires strict"):
        Executor(config)


def test_qualified_pipeline_email_contains_evidence(config, monkeypatch):
    paper = make_sample_paper(publication=dict(EVIDENCE))
    sent = stub_dependencies(monkeypatch, [paper])
    Executor(strict(config)).run()
    assert len(sent) == 1
    body = message_from_string(sent[0][2]).get_payload(decode=True).decode("utf-8")
    assert "CVPR" in body and "2025" in body and "视觉检测" in body
    assert EVIDENCE["verification_url"] in body
    assert EVIDENCE["publication_url"] in body


def test_all_rejected_sends_no_email(config, monkeypatch):
    sent = stub_dependencies(monkeypatch, [])
    Executor(strict(config)).run()
    assert sent == []


def test_email_escapes_untrusted_metadata():
    paper = make_sample_paper(title='<script>alert(1)</script>', tldr='<img src=x onerror=alert(1)>',
                              pdf_url='javascript:alert(1)', publication=dict(EVIDENCE))
    html = render_email([paper])
    assert "<script>" not in html and "<img src=x" not in html
    assert "javascript:" not in html
    assert "&lt;script&gt;" in html
