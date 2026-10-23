"""One-off mail safety: preview, exact recipient, TLS, and ambiguous delivery."""

import importlib.util
import json
from pathlib import Path
import smtplib
import ssl
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("send_curated", ROOT / "scripts/send_curated.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
MANIFEST = ROOT / "selections/post-training-2026-10.json"
ENV = {"SENDER": "sender@example.com", "RECEIVER": "recipient@example.com", "SENDER_PASSWORD": "test-only"}


class FakeSMTP:
    def __init__(self, *args, **kwargs):
        self.calls = []
        self.connection = (args, kwargs)
        self.send_error = None
        self.quit_error = None
        self.refused = {}

    def login(self, sender, password):
        self.calls.append(("login", sender, password))

    def send_message(self, message, **kwargs):
        self.calls.append(("send", message, kwargs))
        if self.send_error:
            raise self.send_error
        return self.refused

    def quit(self):
        self.calls.append(("quit",))
        if self.quit_error:
            raise self.quit_error

    def close(self):
        self.calls.append(("close",))


def selection():
    return MODULE.load_selection(MANIFEST)


def submit(tmp_path, smtp, env=None):
    selected = selection()
    plain, html = MODULE.render(selected)
    return MODULE.submit(selected, plain, html, tmp_path / "receipt.json", environ=env or ENV, smtp_factory=smtp)


def test_default_cli_only_renders_without_secrets(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "scripts/send_curated.py"), "--manifest", str(MANIFEST),
                             "--output-dir", str(tmp_path)], capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["smtp_accepted"] is False
    assert "DeepSeek-R1" in (tmp_path / "email.html").read_text(encoding="utf-8")


def test_send_uses_tls_exact_recipient_and_multipart_content(tmp_path):
    instances = []

    def factory(*args, **kwargs):
        server = FakeSMTP(*args, **kwargs)
        instances.append(server)
        return server

    receipt = submit(tmp_path, factory)
    assert len(instances) == 1
    server = instances[0]
    assert server.connection[0] == ("smtp.163.com", 465)
    assert server.connection[1]["timeout"] == 30
    context = server.connection[1]["context"]
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname
    sends = [call for call in server.calls if call[0] == "send"]
    assert len(sends) == 1
    message, envelope = sends[0][1:]
    assert envelope == {"from_addr": ENV["SENDER"], "to_addrs": [ENV["RECEIVER"]]}
    assert str(message["Subject"]) == selection()["subject"]
    assert message.get_body(preferencelist=("html",)).get_content().count("正式研究论文") == 1
    assert "Nature" in message.get_body(preferencelist=("plain",)).get_content()
    assert receipt["smtp_accepted"] is True
    serialized = (tmp_path / "receipt.json").read_text(encoding="utf-8")
    assert ENV["SENDER_PASSWORD"] not in serialized and ENV["RECEIVER"] not in serialized


@pytest.mark.parametrize("issue", [TimeoutError("uncertain"), smtplib.SMTPServerDisconnected("uncertain")])
def test_delivery_uncertainty_does_not_retry(tmp_path, issue):
    server = FakeSMTP()
    server.send_error = issue
    with pytest.raises(RuntimeError, match="inspect receipt"):
        submit(tmp_path, lambda *args, **kwargs: server)
    assert len([c for c in server.calls if c[0] == "send"]) == 1
    receipt = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "delivery_uncertain" and receipt["smtp_accepted"] is False


def test_quit_failure_after_acceptance_is_still_success(tmp_path):
    server = FakeSMTP()
    server.quit_error = smtplib.SMTPServerDisconnected("quit")
    receipt = submit(tmp_path, lambda *args, **kwargs: server)
    assert receipt["status"] == "accepted_by_smtp" and receipt["smtp_accepted"] is True
    assert receipt["connection_close_warning"] == "SMTPServerDisconnected"


def test_refused_recipient_is_not_success(tmp_path):
    server = FakeSMTP()
    server.refused = {ENV["RECEIVER"]: (550, b"rejected")}
    with pytest.raises(RuntimeError):
        submit(tmp_path, lambda *args, **kwargs: server)
    receipt = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))
    assert not receipt["smtp_accepted"]


def test_rerun_blocked_before_connection(tmp_path):
    with pytest.raises(ValueError, match="re-runs"):
        submit(tmp_path, lambda *args, **kwargs: pytest.fail("must not connect"), {**ENV, "GITHUB_RUN_ATTEMPT": "2"})


@pytest.mark.parametrize("mailbox", ["a@example.com,b@example.com", "Name <a@example.com>", "a@example.com\r\nBcc: other@example.com", None])
def test_invalid_recipient_rejected_before_connection(tmp_path, mailbox):
    with pytest.raises(ValueError):
        submit(tmp_path, lambda *args, **kwargs: pytest.fail("must not connect"), {**ENV, "RECEIVER": mailbox})


def test_manifest_rejects_unofficial_or_unsafe_link(tmp_path):
    selected = selection()
    selected["papers"][0]["publication_url"] = "https://www.nature.com.evil.example/paper"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(selected), encoding="utf-8")
    with pytest.raises(ValueError, match="official HTTPS"):
        MODULE.load_selection(path)


def test_render_escapes_content():
    selected = selection()
    selected["papers"][0]["core"] = '<script>alert("unsafe")</script>'
    _, html = MODULE.render(selected)
    assert "<script>" not in html and "&lt;script&gt;" in html
