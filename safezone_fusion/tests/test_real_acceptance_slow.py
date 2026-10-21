"""Optional slow regression: run real recorded-camera clips through the full
SafeZone pipeline. Skipped by default.

Run explicitly:

    pytest projects/safezone_fusion/tests -m slow -q
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from safezone_fusion.tools.run_real_acceptance import run_acceptance

@pytest.mark.slow
def test_real_acceptance_overall_pass(tmp_path):
    report = run_acceptance(output_dir=tmp_path, print_summary=False)
    failed = [c for c in report["cases"] if not c.get("pass")]
    assert not failed, "real-acceptance failures: " + ", ".join(
        f"{c['name']}({'; '.join(c.get('errors', []))})" for c in failed
    )
    assert report.get("pass") is True
