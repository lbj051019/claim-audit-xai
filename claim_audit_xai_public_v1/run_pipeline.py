#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run the full ClaimAuditXAI pipeline from the command line."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ClaimAuditXAI pipeline.")
    parser.add_argument("--pdf", help="Input PDF for extraction")
    parser.add_argument("--claims-file", default="environmental_claims.txt", help="Text file of claims to audit")
    parser.add_argument("--workdir", default=".", help="Working directory for outputs")
    parser.add_argument("--skip-extraction", action="store_true", help="Skip the PDF extraction step")
    args = parser.parse_args()

    workdir = Path(args.workdir).resolve()
    if not workdir.exists():
        raise FileNotFoundError(f"Workdir not found: {workdir}")

    if not args.skip_extraction:
        if not args.pdf:
            raise SystemExit("--pdf is required unless --skip-extraction is used.")
        run([sys.executable, str(workdir / "extract_claims_sdk.py"), args.pdf])
        run([sys.executable, str(workdir / "claim_filter.py"), "--input-file", "extracted_claims.json", "--output-file", "environmental_claims.txt"])

    run([sys.executable, str(workdir / "claim_audit_xai.py"), "--claims", args.claims_file])


if __name__ == "__main__":
    main()
