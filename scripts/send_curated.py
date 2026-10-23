"""Render an editorially verified selection; send only with an explicit --send."""

import argparse
from datetime import datetime, timezone
from email.message import EmailMessage
from email.utils import format_datetime, parseaddr
import hashlib
from html import escape
import json
import os
from pathlib import Path
import smtplib
import ssl
from urllib.parse import urlsplit


PUBLICATION_HOSTS = {"proceedings.neurips.cc", "papers.nips.cc", "www.nature.com"}


def load_selection(path):
    selection = json.loads(Path(path).read_text(encoding="utf-8"))
    for field in ("batch_id", "subject", "intro", "reading_order"):
        if not isinstance(selection.get(field), str) or not selection[field].strip():
            raise ValueError(f"Missing {field}")
    if any(c in selection["subject"] for c in "\r\n"):
        raise ValueError("Invalid subject")
    papers = selection.get("papers", [])
    if not 1 <= len(papers) <= 10:
        raise ValueError("Expected 1 to 10 papers")
    titles = set()
    for paper in papers:
        for field in ("title", "short_title", "authors", "venue", "core", "why", "focus", "limits"):
            if not isinstance(paper.get(field), str) or not paper[field].strip():
                raise ValueError(f"Missing paper field: {field}")
        if paper.get("topic") != "post_training" or paper.get("publication_kind") not in {"journal", "conference"}:
            raise ValueError("Unexpected paper topic or publication kind")
        if type(paper.get("year")) is not int or not 2000 <= paper["year"] <= datetime.now(timezone.utc).year:
            raise ValueError("Invalid publication year")
        title = paper["title"].casefold().strip()
        if title in titles:
            raise ValueError("Duplicate paper")
        titles.add(title)
        url = urlsplit(paper.get("publication_url", ""))
        if (url.scheme != "https" or url.hostname not in PUBLICATION_HOSTS
                or url.username or url.password or url.port not in (None, 443)
                or not url.path or url.path == "/"):
            raise ValueError("Expected an official HTTPS publication page")
    return selection


def render(selection):
    text = [selection["subject"], "", selection["intro"], "", "建议阅读顺序：" + selection["reading_order"], ""]
    cards = []
    for number, paper in enumerate(selection["papers"], 1):
        kind = "顶刊 · 正式研究论文" if paper["publication_kind"] == "journal" else "顶会 · 主会论文"
        metadata = f'{paper["venue"]} {paper["year"]} · {kind}'
        lines = [("核心内容", paper["core"]), ("推荐理由", paper["why"]),
                 ("阅读重点", paper["focus"]), ("适用边界", paper["limits"])]
        text.extend([f'{number}. {paper["title"]}', metadata, paper["authors"]])
        text.extend(f"{label}：{value}" for label, value in lines)
        text.extend(["正式出版页面：" + paper["publication_url"], ""])
        details = "".join(f"<p><strong>{label}：</strong>{escape(value)}</p>" for label, value in lines)
        cards.append(
            '<div style="background:#ffffff;border:1px solid #dbe3ec;border-radius:12px;padding:22px;margin:18px 0">'
            f'<p style="color:#475569;font-size:14px">{escape(metadata)}</p>'
            f'<h2 style="font-size:21px;line-height:1.45">{number}. {escape(paper["short_title"])}</h2>'
            f'<p><a href="{escape(paper["publication_url"], quote=True)}" style="color:#1d4ed8">{escape(paper["title"])}</a></p>'
            f'<p style="color:#64748b;font-size:14px">{escape(paper["authors"])}</p>{details}</div>'
        )
    html = ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"></head>'
            '<body style="margin:0;background:#f1f5f9;color:#172033;font-family:Arial,sans-serif;line-height:1.8">'
            '<div style="max-width:760px;margin:auto;padding:28px 16px">'
            f'<h1 style="font-size:26px">{escape(selection["subject"])}</h1>'
            f'<p>{escape(selection["intro"])}</p>'
            f'<p><strong>建议阅读顺序：</strong>{escape(selection["reading_order"])}</p>'
            + "".join(cards) + '</div></body></html>')
    return "\n".join(text), html


def _mailbox(value):
    if not isinstance(value, str) or any(c in value for c in "\r\n"):
        raise ValueError("Invalid mailbox configuration")
    name, address = parseaddr(value)
    if name or address != value or address.count("@") != 1 or not all(address.split("@")):
        raise ValueError("Expected one bare mailbox address")
    return address


def submit(selection, plain, html, output, environ=None, smtp_factory=None):
    env = os.environ if environ is None else environ
    smtp_factory = smtplib.SMTP_SSL if smtp_factory is None else smtp_factory
    if env.get("GITHUB_RUN_ATTEMPT", "1") != "1":
        raise ValueError("Refusing workflow re-runs to avoid duplicate email")
    sender = _mailbox(env.get("SENDER"))
    receiver = _mailbox(env.get("RECEIVER"))
    password = env.get("SENDER_PASSWORD")
    if not password:
        raise ValueError("Missing sender password")
    digest = hashlib.sha256(json.dumps(selection, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]
    message = EmailMessage()
    message["From"] = sender
    message["To"] = receiver
    message["Subject"] = selection["subject"]
    message["Date"] = format_datetime(datetime.now(timezone.utc))
    message["Message-ID"] = f"<curated-{digest}@{sender.split('@')[1]}>"
    message.set_content(plain)
    message.add_alternative(html, subtype="html")
    receipt = {"batch_id": selection["batch_id"], "paper_count": len(selection["papers"]),
               "subject": selection["subject"], "status": "connecting", "smtp_accepted": False}

    def record():
        receipt["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")

    record()
    server = None
    submitting = False
    try:
        server = smtp_factory("smtp.163.com", 465, timeout=30, context=ssl.create_default_context())
        server.login(sender, password)
        receipt["status"] = "submitting"
        record()
        submitting = True
        refused = server.send_message(message, from_addr=sender, to_addrs=[receiver])
        if refused:
            raise smtplib.SMTPRecipientsRefused(refused)
        receipt.update(status="accepted_by_smtp", smtp_accepted=True)
        record()  # Persist success before QUIT; a QUIT failure must not trigger a retry.
    except Exception as exc:
        receipt["status"] = "delivery_uncertain" if submitting else "failed_before_submission"
        receipt["error_type"] = type(exc).__name__
        record()
        raise RuntimeError("Email submission did not finish normally; inspect receipt before considering any retry") from None
    finally:
        if server is not None:
            try:
                server.quit()
            except Exception as exc:
                receipt["connection_close_warning"] = type(exc).__name__
                record()
                try:
                    server.close()
                except Exception:
                    pass
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--send", action="store_true")
    args = parser.parse_args()
    selection = load_selection(args.manifest)
    plain, html = render(selection)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "email.html").write_text(html, encoding="utf-8")
    (output / "email.txt").write_text(plain, encoding="utf-8")
    if args.send:
        result = submit(selection, plain, html, output / "receipt.json")
    else:
        result = {"status": "preview_only", "smtp_accepted": False, "paper_count": len(selection["papers"])}
        (output / "receipt.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
