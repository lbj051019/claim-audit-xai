from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


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

    repo_dir = Path(__file__).resolve().parent

    if not args.skip_extraction:
        if not args.pdf:
            raise SystemExit("--pdf is required unless --skip-extraction is used.")
        run([sys.executable, str(repo_dir / "extract_claims_sdk.py"), args.pdf], cwd=workdir)
        run([sys.executable, str(repo_dir / "claim_filter.py"), "--input-file", "extracted_claims.json", "--output-file", args.claims_file], cwd=workdir)

    run([sys.executable, str(repo_dir / "main.py"), "--claims-file", args.claims_file], cwd=workdir)


if __name__ == "__main__":
    main()
