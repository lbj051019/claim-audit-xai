from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

from claims_pipeline import evaluate_claims
from config import URLS
from crawler import fetch_dynamic, fetch_govuk_content, fetch_pdf
from processor import chunk_text, clean_text
from rag import RAG
from report_generator import generate_report


def load_claims(file_path: str) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return [line.strip("- ").strip() for line in lines if line.strip()]


def collect_reference_text() -> List[str]:
    all_text: List[str] = []

    text, links = fetch_govuk_content(URLS["CMA_HTML"])
    all_text.append(text)

    for link in links[:2]:
        sub_text, _ = fetch_govuk_content(link)
        all_text.append(sub_text)

    all_text.append(fetch_pdf(URLS["CMA_PDF"]))

    mmo_text, _ = fetch_govuk_content(URLS["MMO"])
    all_text.append(mmo_text)

    ohi_text = fetch_dynamic(URLS["OHI"])
    if ohi_text:
        all_text.append(ohi_text)

    return all_text


def build_rag_index() -> RAG:
    print("Collecting official source text...")
    raw_texts = collect_reference_text()
    cleaned = [clean_text(text) for text in raw_texts if text]

    chunks = []
    for text in cleaned:
        chunks.extend(chunk_text(text))

    print(f"Collected {len(cleaned)} source documents and built {len(chunks)} chunks.")

    rag = RAG()
    rag.build(chunks)
    return rag


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG + XAI evaluation stage.")
    parser.add_argument("--claims-file", default="environmental_claims.txt", help="Text file containing one claim per line")
    parser.add_argument("--report-txt", default="report.txt", help="Output TXT report path")
    parser.add_argument("--report-json", default="report.json", help="Output JSON report path")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")
    if not Path(args.claims_file).exists():
        raise FileNotFoundError(f"Claims file not found: {args.claims_file}")

    rag = build_rag_index()
    claims = load_claims(args.claims_file)
    if not claims:
        raise ValueError("No claims found in the claims file.")

    results = evaluate_claims(claims, rag)
    generate_report(results, output_txt=args.report_txt, output_json=args.report_json)

    for result in results:
        print("\n====================")
        print("Claim:", result["claim"])
        print("Support Level:", result["support_level"])
        print("Score:", result["score"])
        print("Verdict:", result["verdict"])
        print("Evidence:", result["evidence"][:200])
        print("Explanation:")
        for item in result["explanation"]:
            print("-", item)


if __name__ == "__main__":
    main()
