"""Conservative topic and formal-publication gate for the daily digest.

The lexical topic rules intentionally prefer false negatives to unrelated mail.
DBLP supplies bibliographic evidence; an arXiv ``journal_ref`` or comment never
does. No paper body, model, Zotero library, or mail service is accessed here.
"""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
import html
import math
import re
import time
import unicodedata
from urllib.parse import urlsplit

import requests


SEARCH_ENDPOINT = "https://dblp.org/search/publ/api"
SEARCH_ENDPOINTS = frozenset(
    (SEARCH_ENDPOINT, "https://dblp.uni-trier.de/search/publ/api")
)


def _words(value):
    """Normalize typography without fuzzy matching or dropping title words."""
    if not isinstance(value, str):
        return ""
    value = html.unescape(value)
    value = re.sub(r"</?(?:i|b|em|strong|sub|sup|span)(?:\s[^>]*)?>", "", value,
                   flags=re.IGNORECASE)
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"\w+|[+<=>]", value))


def _author_name(value):
    if not isinstance(value, str):
        return ""
    value = re.sub(r"\s+\d{4}$", "", value.strip())  # DBLP name disambiguator
    value = unicodedata.normalize("NFKD", value)
    value = "".join(c for c in value if not unicodedata.combining(c))
    name = _words(value)
    return "" if name in {"", "unknown", "anonymous", "et al"} else name


def _get(value, key, default=None):
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def _authors(value):
    if isinstance(value, Mapping) and "author" in value:
        value = value["author"]
    if not isinstance(value, (list, tuple)):
        value = [value]
    result = set()
    for author in value:
        name = author if isinstance(author, str) else _get(author, "name", _get(author, "text", ""))
        normalized = _author_name(name)
        if normalized:
            result.add(normalized)
    return result


@dataclass(frozen=True)
class _Venue:
    prefix: str
    aliases: tuple[str, ...]
    journal: bool = False


# IDs are deliberately explicit. An unknown config value must never widen scope.
# NMI's DBLP key is journals/natmi, not journals/nmi.
VENUES = {
    "TPAMI": _Venue("journals/pami/", ("IEEE Trans. Pattern Anal. Mach. Intell.",
        "IEEE Transactions on Pattern Analysis and Machine Intelligence", "TPAMI"), True),
    "IJCV": _Venue("journals/ijcv/", ("Int. J. Comput. Vis.",
        "International Journal of Computer Vision", "IJCV"), True),
    "JMLR": _Venue("journals/jmlr/", ("J. Mach. Learn. Res.",
        "Journal of Machine Learning Research", "JMLR"), True),
    "AIJ": _Venue("journals/ai/", ("Artif. Intell.", "Artificial Intelligence"), True),
    "TIP": _Venue("journals/tip/", ("IEEE Trans. Image Process.",
        "IEEE Transactions on Image Processing", "TIP"), True),
    "NMI": _Venue("journals/natmi/", ("Nat. Mac. Intell.", "Nat. Mach. Intell.",
        "Nature Machine Intelligence"), True),
    "CVPR": _Venue("conf/cvpr/", ("CVPR", "IEEE/CVF Conference on Computer Vision and Pattern Recognition",
        "IEEE Conference on Computer Vision and Pattern Recognition")),
    "ICCV": _Venue("conf/iccv/", ("ICCV", "IEEE/CVF International Conference on Computer Vision",
        "IEEE International Conference on Computer Vision")),
    "ECCV": _Venue("conf/eccv/", ("ECCV", "European Conference on Computer Vision")
        + tuple(f"ECCV ({part})" for part in range(1, 101))),
    "NEURIPS": _Venue("conf/nips/", ("NeurIPS", "NIPS", "Advances in Neural Information Processing Systems")),
    "ICML": _Venue("conf/icml/", ("ICML", "International Conference on Machine Learning")),
    "ICLR": _Venue("conf/iclr/", ("ICLR", "International Conference on Learning Representations")),
    "ACL": _Venue("conf/acl/", ("ACL", "ACL (1)", "ACL (2)",
        "Annual Meeting of the Association for Computational Linguistics")),
    "EMNLP": _Venue("conf/emnlp/", ("EMNLP", "Conference on Empirical Methods in Natural Language Processing")),
}

_LM = re.compile(r"\b(?:llms?|vlms?|mllms?|(?:large |multimodal |vision |visual )*language models?)\b")
_POST = re.compile(
    r"\b(?:post training|supervised fine tuning|instruction (?:fine tuning|tuning)|"
    r"(?:direct |group relative )?preference (?:optimization|learning|alignment)|"
    r"reinforcement learning (?:from (?:human|ai) feedback|with verifiable rewards)|"
    r"training (?:large )?language models? to follow instructions|"
    r"reinforcement fine tuning|rlhf|rlaif|dpo|rlvr|grpo|sft|"
    r"reward (?:model(?:s|ing)?|learning|optimization))\b"
)
_PTQ = re.compile(r"\b(?:post training quantization|ptq)\b")
_VISUAL_TASK = re.compile(
    r"\b(?:(?:(?:small |tiny |oriented |rotated |salient |3d |2d )?objects?|"
    r"pedestrians?|vehicles?|faces?|defects?) (?:detection|detectors?|localization)|"
    r"(?:visual |image |video |industrial )?anomaly (?:detection|localization)|"
    r"detecting (?:objects|pedestrians|vehicles|faces|defects|visual anomalies))\b"
)
_OBJECT_TASK = re.compile(r"\b(?:objects?|pedestrians?|vehicles?|faces?) (?:detection|detectors?|localization)\b")
_VISION = re.compile(r"\b(?:visual|vision|images?|videos?|pixels?|cameras?|photographs?|"
                     r"bounding boxes|yolo\w*|detr|surface inspection)\b")
_NONVISUAL = re.compile(r"\b(?:network intrusion|intrusion detection|network traffic|"
                        r"cybersecurity|cyber security|text anomalies|textual anomalies|"
                        r"text anomaly|textual anomaly|log anomaly|log anomalies|"
                        r"time series|fraud detection|source code|program analysis|program repair|"
                        r"software (?:bugs?|defects?|vulnerabilities|testing|engineering)|"
                        r"(?:bugs?|defects?|vulnerabilities) in (?:software|programs)|"
                        r"code defects?|code vulnerabilities|bug detection)\b")
_RESEARCH_LEAD = re.compile(r"\b(?:we (?:propose|present|introduce|develop|study|investigate|"
                            r"train|optimize|align|address)|this (?:paper|work|study) "
                            r"(?:proposes|presents|introduces|studies|investigates|addresses)|"
                            r"our (?:method|framework|approach) (?:addresses|targets|performs))\b")
_INCIDENTAL = re.compile(r"\b(?:baseline|benchmarks?|evaluate(?:d|s)? on|"
                         r"evaluation on|downstream tasks?|including|such as|applications? to|"
                         r"compar(?:e|es|ed|ing) (?:it |them |ours |our method )?(?:with|to|against))\b")
_BASELINE_SUFFIX = re.compile(r"^(?:\s+\w+){0,3}\s+(?:baselines?|benchmarks?)\b")


def _focused_texts(title, summary, pattern):
    """Return task-bearing title/objective sentences, not benchmark name lists."""
    normalized_title = _words(title)
    if pattern.search(normalized_title):
        yield normalized_title
    for sentence in re.split(r"[.!?]\s+|[\r\n]+", summary):
        sentence = _words(sentence)
        match = pattern.search(sentence)
        lead = _RESEARCH_LEAD.search(sentence[:220])
        if match and lead and lead.start() < match.start() < lead.end() + 300:
            if (not _INCIDENTAL.search(sentence[lead.end():match.start()])
                    and not _BASELINE_SUFFIX.search(sentence[match.end():])):
                yield sentence


def match_topics(title, summary):
    """Conservative, deterministic topic IDs; authors cannot trigger a topic."""
    if not isinstance(title, str) or not isinstance(summary, str):
        return []
    combined = _words(title + " " + summary)
    topics = []
    normalized_title = _words(title)
    if _LM.search(combined) and not _PTQ.search(normalized_title):
        # An abstract's robot reward model does not become LLM post-training
        # merely because another sentence mentions language models.
        if any(not _PTQ.search(text) and _LM.search(text)
               for text in _focused_texts(title, summary, _POST)):
            topics.append("post_training")
    if not _NONVISUAL.search(_words(title)):
        for text in _focused_texts(title, summary, _VISUAL_TASK):
            if not _NONVISUAL.search(text) and (_OBJECT_TASK.search(text) or _VISION.search(combined)):
                # Rendering network/time-series data as an image is still not a visual-detection task.
                if not _NONVISUAL.search(combined):
                    topics.append("visual_detection")
                    break
    return topics


_DISALLOWED = re.compile(r"(?:workshop|findings|\bcorr\b|submitted|under[ _-]?review|"
                         r"preprint|companion|demonstrations|student research)", re.IGNORECASE)
_JOURNAL_TYPES = frozenset(("Articles", "Journal Articles"))


def _venue_id(info, allowed):
    key, venue, kind = info.get("key"), info.get("venue"), info.get("type")
    if not all(isinstance(x, str) for x in (key, venue, kind)):
        return None
    if not re.fullmatch(r"(?:journals|conf)/[A-Za-z0-9_-]+/[A-Za-z0-9_.:-]+", key):
        return None
    claims = " ".join(str(info.get(field, "")) for field in
                      ("key", "venue", "status", "note"))
    if _DISALLOWED.search(claims):
        return None
    for identifier in allowed:
        rule = VENUES[identifier]
        expected_types = _JOURNAL_TYPES if rule.journal else {"Conference and Workshop Papers"}
        # The DBLP conference type includes the word Workshop; inspect venue/key
        # separately above, not that fixed type label.
        if (key.startswith(rule.prefix) and kind in expected_types
                and _words(venue) in {_words(alias) for alias in rule.aliases}):
            return identifier
    return None


def _publication_url(value, venue):
    if not isinstance(value, str) or any(ord(c) < 33 for c in value) or "\\" in value:
        return None
    try:
        parsed = urlsplit(value)
        if (parsed.scheme != "https" or parsed.username or parsed.password
                or parsed.port not in (None, 443)):
            return None
    except ValueError:
        return None
    host, path = parsed.hostname, parsed.path
    if not host or _DISALLOWED.search(path) or re.search(r"(?:cvpr|iccv)\d*w\b", path, re.I):
        return None
    if host in {"doi.org", "dx.doi.org"}:
        if (re.fullmatch(r"/10\.\d{4,9}/\S+", path, re.I)
                and not re.match(r"/10\.(?:48550|1101|20944|21203|2139)/", path)
                and "arxiv" not in path.casefold()):
            return value
        return None
    accepted = False
    if venue in {"TPAMI", "TIP", "CVPR", "ICCV"}:
        accepted |= host == "ieeexplore.ieee.org" and bool(re.fullmatch(r"/document/\d+/?", path))
        accepted |= host in {"computer.org", "www.computer.org"} and bool(re.match(r"/csdl/(?:journal|proceedings)/.+/.+", path))
    if venue in {"IJCV", "ECCV"}:
        accepted |= host == "link.springer.com" and bool(re.match(r"/(?:article|chapter)/10\.1007/.+", path))
    if venue == "JMLR":
        accepted |= host in {"jmlr.org", "www.jmlr.org"} and bool(re.match(r"/papers/(?:v\d+|volume\d+)/[^/]+(?:\.html|\.pdf)$", path))
    if venue == "AIJ":
        accepted |= host in {"sciencedirect.com", "www.sciencedirect.com"} and bool(re.match(r"/science/article/(?:pii|abs/pii)/[A-Za-z0-9]+", path))
        accepted |= host == "linkinghub.elsevier.com" and bool(re.match(r"/retrieve/pii/[A-Za-z0-9]+", path))
    if venue == "NMI":
        accepted |= host in {"nature.com", "www.nature.com"} and bool(re.fullmatch(r"/articles/[A-Za-z0-9.-]+", path))
    if venue in {"CVPR", "ICCV"}:
        accepted |= host == "openaccess.thecvf.com" and bool(re.match(r"/content(?:/|_).+/(?:html|papers)/[^/]+\.(?:html|pdf)$", path))
    if venue == "NEURIPS":
        accepted |= host in {"papers.nips.cc", "papers.neurips.cc", "proceedings.neurips.cc"} and bool(re.match(r"/(?:paper|paper_files/paper)/\d{4}/.+", path))
    if venue == "ICML":
        accepted |= host == "proceedings.mlr.press" and bool(re.fullmatch(r"/v\d+/[^/]+\.(?:html|pdf)", path))
    if venue == "ICLR":
        accepted |= host == "openreview.net" and path in {"/forum", "/pdf"} and bool(re.fullmatch(r"id=[A-Za-z0-9_-]+", parsed.query))
    if venue in {"ACL", "EMNLP"}:
        modern = "acl" if venue == "ACL" else "emnlp"
        legacy = "P" if venue == "ACL" else "D"
        accepted |= host == "aclanthology.org" and bool(re.fullmatch(
            rf"/(?:\d{{4}}\.{modern}-(?:long|short)\.\d+|{legacy}\d{{2}}-\d{{4}})(?:/|\.pdf)?", path))
    return value if accepted else None


def _electronic_editions(value):
    values = value if isinstance(value, (list, tuple)) else [value]
    for entry in values:
        if isinstance(entry, str):
            yield entry
        elif isinstance(entry, Mapping) and isinstance(entry.get("text"), str):
            yield entry["text"]


class CurationGate:
    """Filter raw candidates before full-text conversion.

    Disabled gates return ``{}`` for compatibility; rejection is exactly ``None``.
    An optional ``lookup(title, max_hits)`` hook returns the DBLP search JSON shape
    and owns its network timeout/retry policy. All evidence checks remain here.
    """

    def __init__(self, config=None, *, session=None, sleep_fn=None, lookup=None):
        config = {} if config is None else config
        if not callable(getattr(config, "get", None)):
            raise ValueError("curation config must be a mapping")
        self.enabled = config.get("enabled", False)
        if not isinstance(self.enabled, bool):
            raise ValueError("curation.enabled must be a boolean")
        allowed = config.get("allowed_venues", tuple(VENUES))
        if isinstance(allowed, (str, bytes, Mapping)):
            raise ValueError("curation.allowed_venues must be a list of explicit venue IDs")
        try:
            allowed = tuple(allowed)
        except TypeError as error:
            raise ValueError("curation.allowed_venues must be a list") from error
        if not allowed or any(not isinstance(item, str) or item not in VENUES for item in allowed):
            raise ValueError("curation.allowed_venues contains an unknown ID or is empty")
        self.allowed_venues = tuple(dict.fromkeys(allowed))
        self.timeout_seconds = self._number(config, "timeout_seconds", 10, 0.1, 60)
        self.max_retries = self._number(config, "max_retries", 1, 0, 2, integer=True)
        self.retry_backoff_seconds = self._number(config, "retry_backoff_seconds", 0.5, 0, 5)
        self.max_hits = self._number(config, "max_hits", 30, 1, 100, integer=True)
        self.endpoint = config.get("endpoint", SEARCH_ENDPOINT)
        if self.endpoint not in SEARCH_ENDPOINTS:
            raise ValueError("curation.endpoint must be an official DBLP search endpoint")
        if lookup is not None and not callable(lookup):
            raise ValueError("curation lookup must be callable")
        self._session = session
        self._sleep = sleep_fn if sleep_fn is not None else time.sleep
        self._lookup = lookup
        self._counts = Counter({key: 0 for key in (
            "seen", "disabled", "qualified", "topic_rejected", "invalid_metadata",
            "publication_unverified", "lookup_failed", "requests", "retries")})

    @staticmethod
    def _number(config, key, default, low, high, integer=False):
        value = config.get(key, default)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not low <= value <= high
                or (integer and not isinstance(value, int))):
            raise ValueError(f"curation.{key} must be {'an integer' if integer else 'a number'} in [{low}, {high}]")
        return value

    @property
    def report(self):
        return dict(self._counts)

    def _hits(self, payload):
        if not isinstance(payload, Mapping) or not isinstance(payload.get("result"), Mapping):
            raise ValueError("invalid DBLP response")
        hits = payload["result"].get("hits")
        if not isinstance(hits, Mapping):
            raise ValueError("invalid DBLP hits")
        records = hits.get("hit")
        if records is None:
            if str(hits.get("@total", "")) == "0":
                return []
            raise ValueError("missing DBLP hits")
        if isinstance(records, Mapping):
            records = [records]
        if not isinstance(records, list):
            raise ValueError("invalid DBLP hit list")
        return records[:self.max_hits]

    def _search(self, title):
        if self._lookup is not None:
            try:
                return self._hits(self._lookup(title, self.max_hits))
            except Exception:
                # Injected adapters can use a different HTTP library; failures
                # still cannot turn into publication evidence.
                self._counts["lookup_failed"] += 1
                return None
        if self._session is None:
            self._session = requests.Session()
        for attempt in range(self.max_retries + 1):
            retryable = True
            try:
                self._counts["requests"] += 1
                response = self._session.get(
                    self.endpoint,
                    params={"q": _words(title), "format": "json", "h": self.max_hits, "c": 0},
                    timeout=(min(5, self.timeout_seconds), self.timeout_seconds),
                    allow_redirects=False,
                )
                if response.status_code == 200:
                    return self._hits(response.json())
                retryable = response.status_code in {202, 408, 429} or 500 <= response.status_code < 600
            except (requests.RequestException, ValueError, TypeError):
                pass
            if not retryable or attempt == self.max_retries:
                break
            self._counts["retries"] += 1
            self._sleep(self.retry_backoff_seconds * (attempt + 1))
        self._counts["lookup_failed"] += 1
        return None

    def qualify(self, raw_paper):
        self._counts["seen"] += 1
        if not self.enabled:
            self._counts["disabled"] += 1
            return {}
        title = _get(raw_paper, "title", "")
        summary = _get(raw_paper, "summary", _get(raw_paper, "abstract", ""))
        if not isinstance(title, str) or not 1 <= len(title.strip()) <= 2000 or not isinstance(summary, str):
            self._counts["invalid_metadata"] += 1
            return None
        topics = match_topics(title, summary)
        if not topics:
            self._counts["topic_rejected"] += 1
            return None
        authors = _authors(_get(raw_paper, "authors", []))
        if not authors:
            self._counts["invalid_metadata"] += 1
            return None
        hits = self._search(title)
        if hits is None:
            return None
        for hit in hits:
            if not isinstance(hit, Mapping) or not isinstance(hit.get("info"), Mapping):
                continue
            info = hit["info"]
            if _words(info.get("title")) != _words(title):
                continue
            if not authors.intersection(_authors(info.get("authors", []))):
                continue
            venue = _venue_id(info, self.allowed_venues)
            year = info.get("year")
            if venue is None or isinstance(year, bool) or not re.fullmatch(r"(?:19|20)\d{2}", str(year)):
                continue
            publication_url = next((url for entry in _electronic_editions(info.get("ee"))
                                    if (url := _publication_url(entry, venue))), None)
            if publication_url is None:
                continue
            self._counts["qualified"] += 1
            return {
                "venue": venue,
                "year": int(year),
                "publication_url": publication_url,
                "verification_url": "https://dblp.org/rec/" + info["key"],
                "topics": topics,
                "evidence": "dblp",
            }
        self._counts["publication_unverified"] += 1
        return None
