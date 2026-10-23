"""Offline acceptance tests for topic and DBLP publication identity checks."""

from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import requests

from zotero_arxiv_daily.curation import CurationGate, VENUES, match_topics


TITLE = "Direct Preference Optimization for Language Models"


def paper(title=TITLE, summary="We study alignment of large language models.", **kwargs):
    return SimpleNamespace(title=title, summary=summary,
                           authors=kwargs.pop("authors", [SimpleNamespace(name="Alice Example")]),
                           **kwargs)


def publication(**kwargs):
    info = dict(title=TITLE, authors={"author": [{"text": "Alice Example", "@pid": "a/1"}]},
                key="conf/nips/Example24", venue="NeurIPS", year="2024",
                type="Conference and Workshop Papers",
                ee="https://papers.nips.cc/paper_files/paper/2024/hash/abc-Abstract-Conference.html")
    info.update(kwargs)
    return info


def payload(*infos):
    return {"result": {"hits": {"@total": str(len(infos)),
                                "hit": [{"info": info} for info in infos]}}}


class Response:
    def __init__(self, data=None, status=200):
        self.data = data
        self.status_code = status

    def json(self):
        if isinstance(self.data, Exception):
            raise self.data
        return self.data


class Session:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        assert self.responses, "unexpected additional request"
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture(autouse=True)
def forbid_real_network(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("curation tests must never access the network")
    monkeypatch.setattr(requests.Session, "request", blocked)


def gate_with(*infos, **config):
    session = Session(Response(payload(*infos)))
    return CurationGate({"enabled": True, **config}, session=session, sleep_fn=lambda _: None), session


def test_disabled_compatibility_does_not_query_or_require_metadata():
    gate = CurationGate({"enabled": False})
    assert gate.qualify(None) == {}
    assert gate.report["disabled"] == 1
    assert gate.report["requests"] == 0


def test_omegaconf_and_evidence_contract():
    session = Session(Response(payload(publication())))
    gate = CurationGate(OmegaConf.create({"enabled": True, "allowed_venues": ["NEURIPS"]}),
                        session=session)
    assert gate.qualify(paper()) == {
        "venue": "NEURIPS", "year": 2024, "publication_url": publication()["ee"],
        "verification_url": "https://dblp.org/rec/conf/nips/Example24",
        "topics": ["post_training"], "evidence": "dblp",
    }
    assert gate.report["qualified"] == 1
    report = gate.report
    report["qualified"] = 99
    assert gate.report["qualified"] == 1


@pytest.mark.parametrize("change", [
    {"enabled": "false"}, {"allowed_venues": ["CVPRW"]},
    {"allowed_venues": ["cvpr"]}, {"allowed_venues": "CVPR"},
    {"allowed_venues": []}, {"allowed_venues": None},
    {"timeout_seconds": 0}, {"timeout_seconds": 61}, {"timeout_seconds": float("nan")},
    {"max_retries": 3}, {"max_retries": -1}, {"max_retries": True},
    {"max_hits": 101}, {"max_hits": 1.5}, {"retry_backoff_seconds": 6},
    {"endpoint": "https://dblp.org.evil/search/publ/api"},
])
def test_invalid_configuration_does_not_silently_relax(change):
    with pytest.raises(ValueError):
        CurationGate({"enabled": True, **change})


@pytest.mark.parametrize(("title", "summary", "expected"), [
    ("Supervised Fine-Tuning of Large Language Models", "", "post_training"),
    ("Instruction Tuning for Vision-Language Models", "", "post_training"),
    ("RLHF for LLMs", "", "post_training"),
    ("RLVR for Language Models", "", "post_training"),
    ("GRPO for VLMs", "", "post_training"),
    ("Reward Modeling for Language Models", "", "post_training"),
    ("Post-training Large Multimodal Language Models", "", "post_training"),
    ("Direct Preference Optimization: Your Language Model is Secretly a Reward Model", "", "post_training"),
    ("Training language models to follow instructions with human feedback", "", "post_training"),
    ("Reward Modeling", "We propose reward modeling for large language models.", "post_training"),
    ("Better Alignment", "We propose direct preference optimization for large language models.", "post_training"),
    ("Open-Vocabulary Object Detection", "", "visual_detection"),
    ("Industrial Defect Detection", "We inspect camera images of metal surfaces.", "visual_detection"),
    ("Visual Anomaly Detection", "", "visual_detection"),
    ("Learning Industrial Anomalies", "We present an image anomaly detection method.", "visual_detection"),
    ("Small Object Detection in Remote Sensing Images", "", "visual_detection"),
    ("Object Detection in Images", "We release software implementing our visual detector.", "visual_detection"),
])
def test_topic_positives(title, summary, expected):
    assert match_topics(title, summary) == [expected]


@pytest.mark.parametrize(("title", "summary"), [
    ("Post-Training Quantization for Large Language Models", "We improve post-training quantization."),
    ("PTQ for LLMs", "We propose supervised fine-tuning to calibrate quantization."),
    ("Fine-Tuning Large Language Models", "We fine-tune language models for classification."),
    ("Efficient Robot Learning", "We propose reward modeling for robot control."),
    ("A General Visual Backbone", "We evaluate on downstream tasks including object detection."),
    ("A General Visual Backbone", "We propose a backbone for image classification, including object detection."),
    ("Efficient LLM Serving", "We evaluate a DPO baseline and SFT models."),
    ("Network Intrusion Detection", "We render network traffic as images for anomaly detection."),
    ("Text Anomaly Detection", "We use a vision model to detect textual anomalies."),
    ("Anomaly Detection in Time Series", "We convert the signals to images."),
    ("Anomaly Detection", "We find anomalous financial transactions."),
    ("Defect Detection in Programs", "We detect defects in source code."),
    ("Semantic Segmentation", "We segment camera images with visual models."),
    ("Learning Robot Policies from Preferences", "Language models have motivated new learning algorithms. We propose a reward model for quadruped locomotion."),
    ("Reward Modeling for Robot Control", "Language models have motivated new learning algorithms. We propose a reward model for quadruped locomotion."),
    ("Code Defect Detection with Images", "We transform program source code into images and detect bugs in software."),
    ("Image Defect Detection", "We propose program analysis that detects defects in source code rendered as images."),
    ("Efficient Pretraining of Large Language Models", "We propose a new pretraining method and compare it with supervised fine-tuning baselines."),
    ("Better Pretraining", "We propose a language model using supervised fine-tuning baselines for evaluation."),
])
def test_topic_false_positives_are_rejected_before_network(title, summary):
    gate = CurationGate({"enabled": True})
    assert gate.qualify(paper(title, summary)) is None
    assert gate.report["topic_rejected"] == 1
    assert gate.report["requests"] == 0


def test_authors_do_not_supply_topic_words():
    gate = CurationGate({"enabled": True})
    raw = paper("A Study of Databases", "We improve query planning.",
                authors=["Object Detection LLM DPO"])
    assert gate.qualify(raw) is None
    assert gate.report["topic_rejected"] == 1


def test_multiple_topics_are_retained():
    assert match_topics("Instruction Tuning VLMs for Object Detection", "") == ["post_training", "visual_detection"]


def test_exact_title_accepts_typography_but_not_nearby_papers():
    gate, _ = gate_with(publication(title="<i>DIRECT</i> Preference Optimization for Language Models."))
    assert gate.qualify(paper()) is not None
    gate, _ = gate_with(publication(title=TITLE + " with Better Rewards"))
    assert gate.qualify(paper()) is None
    gate, _ = gate_with(publication(title=TITLE.replace("Language Models", "Vision Language Models")))
    assert gate.qualify(paper()) is None


def test_meaningful_title_symbols_are_not_dropped():
    gate, _ = gate_with(publication(title=TITLE + " C++"))
    assert gate.qualify(paper(TITLE + " C")) is None


@pytest.mark.parametrize("authors", [[], {"author": "Other Person"}, {"author": {"text": "A. Example"}},
                                     {"author": {"text": "Example"}}])
def test_wrong_or_missing_dblp_authors_are_rejected(authors):
    gate, _ = gate_with(publication(authors=authors))
    assert gate.qualify(paper()) is None


def test_author_diacritics_and_dblp_disambiguation_suffix():
    gate, _ = gate_with(publication(authors={"author": {"text": "Álice Example 0001"}}))
    assert gate.qualify(paper(authors=["Alice Example"])) is not None


@pytest.mark.parametrize("authors", [[], ["Unknown"], ["Anonymous"], [None]])
def test_invalid_raw_authors_fail_without_lookup(authors):
    gate = CurationGate({"enabled": True})
    assert gate.qualify(paper(authors=authors)) is None
    assert gate.report["invalid_metadata"] == 1
    assert gate.report["requests"] == 0


@pytest.mark.parametrize("venue_id", list(VENUES))
def test_each_explicit_venue_has_a_verified_rule(venue_id):
    rule = VENUES[venue_id]
    gate, _ = gate_with(publication(key=rule.prefix + "Example24", venue=rule.aliases[0],
                                   type="Journal Articles" if rule.journal else "Conference and Workshop Papers",
                                   ee="https://doi.org/10.1234/formal-paper"), allowed_venues=[venue_id])
    assert gate.qualify(paper())["venue"] == venue_id


def test_allowed_venues_restricts_other_valid_main_venues():
    gate, _ = gate_with(publication(), allowed_venues=["ICLR"])
    assert gate.qualify(paper()) is None


@pytest.mark.parametrize("change", [
    {"key": "conf/cvprw/Example24", "venue": "CVPR"},
    {"key": "conf/acl/2024.findings-acl.1", "venue": "ACL"},
    {"key": "conf/iclr/workshop-Example24", "venue": "ICLR"},
    {"key": "conf/nips/Example24", "venue": "NeurIPS Workshops"},
    {"key": "journals/corr/abs-2401-00001", "venue": "NeurIPS"},
    {"venue": "ICLR"}, {"type": "Informal Publications"}, {"type": "Editorship"},
    {"type": "Journal Articles"}, {"status": "submitted"}, {"note": "under review"},
    {"venue": "NeurIPS (Datasets and Benchmarks)"}, {"key": "conf/nips/../forged"},
    {"venue": ["NeurIPS"]}, {"key": None},
])
def test_publication_identity_failures_are_closed(change):
    gate, _ = gate_with(publication(**change))
    assert gate.qualify(paper()) is None
    assert gate.report["publication_unverified"] == 1


def test_eccv_numbered_main_proceedings_supported_but_not_workshops():
    gate, _ = gate_with(publication(key="conf/eccv/Example24", venue="ECCV (12)",
                                   ee="https://link.springer.com/chapter/10.1007/978-1-23_4"))
    assert gate.qualify(paper())["venue"] == "ECCV"
    gate, _ = gate_with(publication(key="conf/eccv/Example24", venue="ECCV Workshops (12)"))
    assert gate.qualify(paper()) is None


def test_arxiv_venue_claim_is_not_evidence_and_stale_comment_does_not_override_dblp():
    raw = paper(journal_ref="NeurIPS 2024", comment="submitted, under review")
    gate, _ = gate_with()
    assert gate.qualify(raw) is None
    gate, _ = gate_with(publication())
    assert gate.qualify(raw) is not None


@pytest.mark.parametrize("url", [
    "https://arxiv.org/abs/2401.12345", "https://doi.org/10.48550/arXiv.2401.12345",
    "http://papers.nips.cc/paper/2024/hash/abc.html", "https://papers.nips.cc.evil/paper/2024/x",
    "https://papers.nips.cc/", "https://user:pass@papers.nips.cc/paper/2024/x",
    "https://papers.nips.cc:444/paper/2024/x", "https://papers.nips.cc\\@evil/paper/2024/x",
    "https://doi.org/", "https://doi.org/10.1101/2024.01.01.1234",
    "https://openreview.net/forum?id=formal", "https://nature.com/articles/abc",
    "https://papers.nips.cc/paper/2024/workshops/abc.html", "https://doi.org/10.1234/CVPRW.2024.1",
])
def test_invalid_or_nonformal_publication_urls_are_rejected(url):
    gate, _ = gate_with(publication(ee=url))
    assert gate.qualify(paper()) is None


@pytest.mark.parametrize(("venue_id", "url"), [
    ("TPAMI", "https://ieeexplore.ieee.org/document/123456"),
    ("IJCV", "https://link.springer.com/article/10.1007/s12345-024-00001"),
    ("JMLR", "https://jmlr.org/papers/v25/24-100.html"),
    ("AIJ", "https://www.sciencedirect.com/science/article/pii/S000437022400001"),
    ("NMI", "https://www.nature.com/articles/s42256-024-00001"),
    ("CVPR", "https://openaccess.thecvf.com/content/CVPR2024/html/Example_CVPR_2024_paper.html"),
    ("ICML", "https://proceedings.mlr.press/v235/example24.html"),
    ("ICLR", "https://openreview.net/forum?id=abc-XYZ_123"),
    ("ACL", "https://aclanthology.org/2024.acl-long.123/"),
    ("EMNLP", "https://aclanthology.org/2024.emnlp-short.12/"),
])
def test_official_paper_specific_urls(venue_id, url):
    rule = VENUES[venue_id]
    gate, _ = gate_with(publication(key=rule.prefix + "Example24", venue=rule.aliases[0],
                                   type="Articles" if rule.journal else "Conference and Workshop Papers", ee=url))
    assert gate.qualify(paper())["publication_url"] == url


def test_openreview_requires_approved_iclr_identity_and_paper_id():
    for url in ("https://openreview.net/", "https://openreview.net/forum", "https://openreview.net/group?id=ICLR.cc"):
        gate, _ = gate_with(publication(key="conf/iclr/Example24", venue="ICLR", ee=url))
        assert gate.qualify(paper()) is None


def test_selects_formal_url_from_multiple_electronic_editions():
    gate, _ = gate_with(publication(ee=["https://arxiv.org/abs/2401.12345", {"text": "https://doi.org/10.1234/paper"}]))
    assert gate.qualify(paper())["publication_url"] == "https://doi.org/10.1234/paper"


@pytest.mark.parametrize("change", [{"year": None}, {"year": True}, {"year": "forthcoming"},
                                    {"ee": None}, {"type": None}, {"title": None}])
def test_missing_fields_cannot_form_evidence(change):
    gate, _ = gate_with(publication(**change))
    assert gate.qualify(paper()) is None


def test_single_hit_and_empty_result_shapes():
    single = {"result": {"hits": {"@total": "1", "hit": {"info": publication()}}}}
    gate = CurationGate({"enabled": True}, session=Session(Response(single)))
    assert gate.qualify(paper()) is not None
    empty = {"result": {"hits": {"@total": "0"}}}
    gate = CurationGate({"enabled": True}, session=Session(Response(empty)))
    assert gate.qualify(paper()) is None
    assert gate.report["publication_unverified"] == 1
    assert gate.report["lookup_failed"] == 0


@pytest.mark.parametrize("response", [
    Response(status=429), Response(status=503), Response(status=202),
    Response(ValueError("empty/challenge body")), Response({}), requests.Timeout("timeout"),
    requests.ConnectionError("unreachable"),
])
def test_transient_or_malformed_lookup_failure_is_bounded(response):
    session = Session(response, response, response)
    sleeps = []
    gate = CurationGate({"enabled": True, "max_retries": 2, "retry_backoff_seconds": 0.2},
                        session=session, sleep_fn=sleeps.append)
    assert gate.qualify(paper()) is None
    assert len(session.calls) == 3
    assert sleeps == [0.2, 0.4]
    assert gate.report["lookup_failed"] == 1
    assert gate.report["retries"] == 2
    assert gate.report["qualified"] == 0


def test_nonretryable_response_stops_and_retry_can_recover():
    session = Session(Response(status=403))
    gate = CurationGate({"enabled": True}, session=session)
    assert gate.qualify(paper()) is None
    assert len(session.calls) == 1
    session = Session(Response(status=429), Response(payload(publication())))
    gate = CurationGate({"enabled": True}, session=session, sleep_fn=lambda _: None)
    assert gate.qualify(paper()) is not None
    assert gate.report["retries"] == 1
    assert gate.report["lookup_failed"] == 0


def test_request_has_bounded_parameters_and_no_redirects():
    session = Session(Response(payload(publication())))
    gate = CurationGate({"enabled": True, "timeout_seconds": 3, "max_hits": 5,
                         "endpoint": "https://dblp.uni-trier.de/search/publ/api"}, session=session)
    assert gate.qualify(paper())
    url, kwargs = session.calls[0]
    assert url == "https://dblp.uni-trier.de/search/publ/api"
    assert kwargs == {"params": {"q": TITLE.lower(), "format": "json", "h": 5, "c": 0},
                      "timeout": (3, 3), "allow_redirects": False}


def test_injected_lookup_still_enforces_all_identity_checks():
    calls = []
    def lookup(title, max_hits):
        calls.append((title, max_hits))
        return payload(publication(authors={"author": "Wrong Author"}), publication())
    gate = CurationGate({"enabled": True}, lookup=lookup)
    assert gate.qualify(paper())["evidence"] == "dblp"
    assert calls == [(TITLE, 30)]
    assert gate.report["requests"] == 0  # Adapter owns transport/retries.


def test_injected_lookup_failure_is_closed():
    def lookup(*_):
        raise RuntimeError("adapter lookup failed")
    gate = CurationGate({"enabled": True}, lookup=lookup)
    assert gate.qualify(paper()) is None
    assert gate.report["lookup_failed"] == 1


def test_mapping_raw_paper_supported_and_no_metadata_mutation():
    raw = dict(title=TITLE, abstract="We study large language models.", authors=[{"name": "Alice Example"}])
    original = dict(raw)
    gate, _ = gate_with(publication())
    assert gate.qualify(raw)["venue"] == "NEURIPS"
    assert raw == original
