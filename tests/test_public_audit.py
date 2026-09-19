"""Contributor credit is public; private source references still fail the audit."""

import os
from pathlib import Path
import subprocess
import sys


def test_history_allows_authorship_but_still_scans_commit_messages(tmp_path):
    scanner = Path(__file__).resolve().parents[1] / "scripts/check_public_tree.py"
    repo = tmp_path / "repo"
    repo.mkdir()
    env = {**os.environ, "GIT_AUTHOR_NAME": "Example Creator",
           "GIT_AUTHOR_EMAIL": "creator@example.invalid",
           "GIT_COMMITTER_NAME": "Example Creator",
           "GIT_COMMITTER_EMAIL": "creator@example.invalid"}

    def git(*args):
        return subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=repo,
                              env=env, check=True, capture_output=True)

    git("init", "-b", "main")
    (repo / "module.py").write_text("VALUE = 1\n")
    git("add", "module.py")
    git("commit", "-m", "Initial implementation")
    terms = tmp_path / "identifiers.txt"
    terms.write_text("Example Creator\ncreator@example.invalid\nprivate-source-marker\n")

    def audit():
        return subprocess.run([sys.executable, str(scanner), "--history", "--terms-file", str(terms)],
                              cwd=repo, capture_output=True, text=True)

    assert audit().returncode == 0
    git("commit", "--allow-empty", "-m", "Imported private-source-marker")
    result = audit()
    assert result.returncode == 1 and "private identifier" in result.stdout
