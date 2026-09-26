"""Fail CI when a detect-secrets JSON scan reports potential secrets.

Usage: python -m detect_secrets scan --no-verify > .secrets-report.json
       python scripts/check_secrets.py .secrets-report.json

The scanner's scan command alone exits successfully even when it finds secrets.
Only locations and detector types are printed; secret values are never printed.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys


def main():
    report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
    results = report["results"]
    count = 0
    for filename, findings in sorted(results.items()):
        for finding in findings:
            count += 1
            print(f"{filename}:{finding['line_number']}: {finding['type']}")
    print(f"detect-secrets: {count} potential secret(s)")
    return 1 if count else 0


if __name__ == "__main__":
    sys.exit(main())
