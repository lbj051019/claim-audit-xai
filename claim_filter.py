#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Filter environmental / sustainability claims from a document result file using OpenAI."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

from openai import OpenAI, RateLimitError

DEFAULT_MODEL = "gpt-5-mini"


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_text_from_parse_result(parse_result: Dict[str, Any]) -> str:
    parts: List[str] = []

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in ("text", "content", "markdown") and isinstance(value, str) and value.strip():
                    parts.append(value.strip())
                elif isinstance(value, (dict, list)):
                    walk(value)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(parse_result)
    seen = set()
    out: List[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return "\n\n".join(out).strip()


def extract_text_from_claims_json(data: Any) -> str:
    lines: List[str] = []
    if isinstance(data, dict):
        data = data.get("claims") if isinstance(data.get("claims"), list) else [data]
    if isinstance(data, list):
        for idx, item in enumerate(data, start=1):
            if isinstance(item, dict):
                claim = item.get("claim") or item.get("claim_text") or item.get("text") or ""
                evidence = item.get("evidence") or ""
                page = item.get("page")
                line = f"{idx}. Claim: {claim}"
                if evidence:
                    line += f"\n   Evidence: {evidence}"
                if page is not None:
                    line += f"\n   Page: {page}"
                lines.append(line)
            else:
                lines.append(f"{idx}. Claim: {str(item)}")
    return "\n".join(lines).strip()


def load_input_text(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    lower = path.lower()
    if lower.endswith(".txt"):
        text = read_text_file(path)
        if not text:
            raise ValueError(f"Input text file is empty: {path}")
        return text
    if lower.endswith(".json"):
        data = load_json(path)
        base = os.path.basename(lower)
        if base == "raw_llamaparse_parse_result.json":
            text = extract_text_from_parse_result(data)
            if text:
                return text
            raise ValueError(f"Could not extract text from parse result JSON: {path}")
        if base == "extracted_claims.json":
            text = extract_text_from_claims_json(data)
            if text:
                return text
            raise ValueError(f"Could not extract text from claims JSON: {path}")
        text = extract_text_from_parse_result(data)
        if text:
            return text
        text = extract_text_from_claims_json(data)
        if text:
            return text
        raise ValueError(f"Unsupported or unreadable JSON structure: {path}")
    raise ValueError("Unsupported input file type. Please provide .txt or .json")


def save_output(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")
    print(f"Saved to: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract environmental or sustainability claims from a document result file.")
    parser.add_argument("--input-file", required=True, help="Path to input file (.txt, extracted_claims.json, or raw_llamaparse_parse_result.json)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model name")
    parser.add_argument("--output-file", default="environmental_claims.txt", help="Where to save the result")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    document_text = load_input_text(args.input_file)
    print(f"Using input from: {os.path.abspath(args.input_file)}")

    prompt = (
        "List every environmental or sustainability claim in this document as bullet points.\n\n"
        "Instructions:\n"
        "- Include only claims about environmental impact, climate, emissions, carbon, net zero, energy, waste, recycling, water, biodiversity, sustainability, ESG, green products, environmentally friendly products, environmental compliance, or similar topics.\n"
        "- Do not include unrelated legal, financial, or operational claims.\n"
        "- Keep each bullet concise.\n"
        "- Use only information present in the document.\n"
        "- If there are no such claims, say exactly: \"No environmental or sustainability claims found.\"\n"
    )

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    try:
        response = client.responses.create(
            model=args.model,
            instructions=(
                "You are a precise document analysis assistant. "
                "Extract only environmental or sustainability-related claims from the provided document."
            ),
            input=f"Document:\n\n{document_text}\n\nTask:\n{prompt}",
        )
    except RateLimitError as e:
        if "insufficient_quota" in str(e):
            raise RuntimeError("OpenAI API quota is insufficient. Add billing or more quota.") from e
        raise

    output_text = response.output_text.strip()
    print("\n===== Environmental / Sustainability Claims =====\n")
    print(output_text)
    print()
    save_output(args.output_file, output_text)


if __name__ == "__main__":
    main()
