"""Bounded publication lookups against DBLP's official, daily-synced knowledge graph.

Uses indexed title literals, not a whole-database substring scan. A title change
can therefore cause a false negative; publication identities are never guessed.
"""
import html
import json
import time
import unicodedata
from urllib.parse import urlsplit, urlunsplit

import requests

ENDPOINT = "https://sparql.dblp.org/sparql"
SCHEMA = "https://dblp.org/rdf/schema#"
REC_PREFIX = "https://dblp.org/rec/"
HTTPS_UPGRADE_HOSTS = frozenset({
    "doi.org", "dx.doi.org", "papers.nips.cc", "papers.neurips.cc",
    "proceedings.neurips.cc", "proceedings.mlr.press", "jmlr.org", "www.jmlr.org",
    "openaccess.thecvf.com", "aclanthology.org", "openreview.net",
    "link.springer.com", "ieeexplore.ieee.org", "www.nature.com",
    "www.sciencedirect.com",
})


def title_variants(title):
    title = " ".join(html.unescape(str(title)).split())
    if not title or len(title) > 1000:
        raise ValueError("Publication title is empty or exceeds the lookup bound")
    normalized = unicodedata.normalize("NFKC", title)
    ascii_punctuation = normalized.translate(str.maketrans({"–": "-", "—": "-", "’": "'", "“": '"', "”": '"'}))
    forms = {title, normalized, ascii_punctuation, normalized.title(), normalized.capitalize()}
    return sorted({variant for form in forms for variant in (form.rstrip("."), form.rstrip(".") + ".")})


def build_query(title, max_hits):
    literals = " ".join(json.dumps(value, ensure_ascii=True) for value in title_variants(title))
    return f"""PREFIX dblp: <{SCHEMA}>
SELECT ?p ?title ?venue ?year ?kind ?ee ?author WHERE {{
  {{ SELECT DISTINCT ?p ?title WHERE {{
      VALUES ?title {{ {literals} }}
      ?p dblp:title ?title .
      VALUES ?publicationKind {{ dblp:Article dblp:Inproceedings }}
      ?p a ?publicationKind .
    }} LIMIT {max_hits}
  }}
  VALUES ?kind {{ dblp:Article dblp:Inproceedings }}
  ?p a ?kind ; dblp:publishedIn ?venue ; dblp:yearOfPublication ?year ;
     dblp:documentPage ?ee ; dblp:authoredBy ?person .
  ?person dblp:creatorName ?author .
}} LIMIT 2000"""


def _publication_link(value):
    parsed = urlsplit(value)
    if parsed.scheme == "http" and parsed.hostname in HTTPS_UPGRADE_HOSTS and not parsed.username and not parsed.password:
        return urlunsplit(("https", parsed.netloc, parsed.path, parsed.query, parsed.fragment))
    return value


def convert_response(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("results", {}).get("bindings"), list):
        raise ValueError("DBLP did not return a valid SPARQL result")
    rows = payload["results"]["bindings"]
    if len(rows) >= 2000:
        raise ValueError("DBLP response reached the row bound; refusing incomplete evidence")
    publications = {}
    for row in rows:
        def value(key):
            field = row.get(key, {})
            return field.get("value", "") if isinstance(field, dict) else ""
        record = value("p")
        if not record.startswith(REC_PREFIX):
            continue
        kind = {SCHEMA + "Article": "Journal Articles", SCHEMA + "Inproceedings": "Conference and Workshop Papers"}.get(value("kind"))
        if not kind:
            continue
        identity = (record, value("title"), value("venue"), value("year"), kind)
        info = publications.setdefault(identity, {
            "key": record[len(REC_PREFIX):], "url": record, "title": value("title"),
            "venue": value("venue"), "year": value("year"), "type": kind,
            "authors": {"author": []}, "ee": [],
        })
        author = value("author")
        if author and author not in info["authors"]["author"]:
            info["authors"]["author"].append(author)
        link = _publication_link(value("ee"))
        if link and link not in info["ee"]:
            info["ee"].append(link)
    return {"result": {"hits": {"hit": [{"info": info} for info in publications.values()]}}}


class DblpSparqlLookup:
    def __init__(self, *, session=None, timeout_seconds=15, max_retries=1, retry_backoff_seconds=2, sleep_fn=time.sleep):
        self.session = session if session is not None else requests.Session()
        self.timeout = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.backoff = float(retry_backoff_seconds)
        if not 1 <= self.timeout <= 30 or not 0 <= self.max_retries <= 2 or not 0 <= self.backoff <= 5:
            raise ValueError("DBLP request settings exceed their safety bounds")
        self.sleep = sleep_fn

    def lookup(self, title, max_hits=10):
        if isinstance(max_hits, bool) or not isinstance(max_hits, int) or not 1 <= max_hits <= 50:
            raise ValueError("max_hits must be between 1 and 50")
        query = build_query(title, max_hits)
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(ENDPOINT, params={"query": query, "format": "json"},
                    headers={"Accept": "application/sparql-results+json", "User-Agent": "zotero-arxiv-daily/1.0 (publication verification)"},
                    timeout=self.timeout, allow_redirects=False)
                if response.status_code != 200:
                    raise ValueError(f"DBLP returned HTTP {response.status_code}")
                return convert_response(response.json())
            except (requests.RequestException, ValueError, TypeError, KeyError) as exc:
                if attempt == self.max_retries:
                    raise LookupError("DBLP publication evidence is unavailable") from exc
                self.sleep(self.backoff)
