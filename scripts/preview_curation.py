"""Read-only selection preview: no Zotero access, PDF downloads, LLM calls or email."""
import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    base = Path(__file__).resolve().parents[1] / "config" / "base.yaml"
    config = OmegaConf.merge(OmegaConf.load(base), OmegaConf.load(args.config))
    config.executor.debug = False
    if config.curation.enabled is not True or list(config.executor.source) != ["arxiv"]:
        raise ValueError("Preview requires enabled strict curation and source=[arxiv]")
    retriever = ArxivRetriever(config)
    papers = retriever._retrieve_raw_papers()
    report = {
        "mode": "read_only_preview",
        "email_sent": False,
        "stats": retriever.curation.report,
        "papers": [dict(title=p.title, arxiv_url=p.entry_id, **p._publication_evidence) for p in papers],
    }
    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
