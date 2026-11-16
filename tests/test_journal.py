import subprocess

from howdy360 import journal

LINES = [
    "Login approved",
    "Failure, timeout reached",
    "Failure, timeout reached",
    "Skipped authentication, closed lid detected",
    "Failure, not a single known face",
    "pam_unix(gdm-password:auth): authentication failure",
]


def test_parse_counts_outcomes():
    counts = journal.parse(LINES)
    assert counts == {"approved": 1, "timeout": 2, "skipped": 1, "failed": 1}


def test_read_uses_journalctl(monkeypatch):
    def fake_run(cmd, **kw):
        assert "SYSLOG_IDENTIFIER=pam_howdy" in cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="\n".join(LINES), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert journal.read(3)["timeout"] == 2


def test_read_returns_none_on_error(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "denied"))
    assert journal.read() is None

    def boom(*a, **k):
        raise OSError("no journalctl")

    monkeypatch.setattr(subprocess, "run", boom)
    assert journal.read() is None
