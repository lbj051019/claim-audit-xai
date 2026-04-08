#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Extract claims from a PDF using LlamaExtract and save JSON / CSV outputs."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from typing import Any, List, Optional

from llama_cloud_services import LlamaExtract
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("claim_extraction")


class ClaimItem(BaseModel):
    claim: str = Field(description="A distinct claim identified in the document.")
    evidence: Optional[str] = Field(default=None, description="Short supporting excerpt.")
    page: Optional[int] = Field(default=None, description="Page number if available.")
    confidence: Optional[float] = Field(default=None, description="Confidence between 0 and 1 if available.")


class ClaimsExtraction(BaseModel):
    claims: List[ClaimItem] = Field(description="All important claims identified in the document.")


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    logger.info("Saved JSON -> %s", path)


def save_claims_csv(path: str, claims: List[dict]) -> None:
    cols = ["claim", "evidence", "page", "confidence"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in claims:
            writer.writerow({k: row.get(k, "") for k in cols})
    logger.info("Saved CSV -> %s", path)


def normalize_result_data(result: Any) -> dict:
    if result is None:
        return {}
    for attr in ("model_dump", "dict"):
        if hasattr(result, attr):
            try:
                return getattr(result, attr)()
            except Exception:
                pass
    if isinstance(result, dict):
        return result
    return {"raw_result": str(result)}


def get_or_create_agent(extractor: LlamaExtract, agent_name: str):
    try:
        logger.info("Trying to reuse existing agent: %s", agent_name)
        return extractor.get_agent(name=agent_name)
    except Exception:
        logger.info("Agent not found; creating a new one: %s", agent_name)
    return extractor.create_agent(name=agent_name, data_schema=ClaimsExtraction)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract claims from a PDF using LlamaExtract.")
    parser.add_argument("pdf", help="Path to PDF file")
    parser.add_argument("--agent-name", default="claim-extraction-agent", help="Extraction agent name")
    parser.add_argument("--raw-output", default="llamaextract_sdk_result.json", help="Raw SDK result JSON")
    parser.add_argument("--output-json", default="extracted_claims.json", help="Structured claims JSON")
    parser.add_argument("--output-csv", default="extracted_claims.csv", help="Claims CSV")
    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        logger.error("PDF not found: %s", args.pdf)
        sys.exit(1)
    if not os.getenv("LLAMA_CLOUD_API_KEY"):
        logger.error("LLAMA_CLOUD_API_KEY is not set.")
        sys.exit(1)

    try:
        extractor = LlamaExtract()
        agent = get_or_create_agent(extractor, args.agent_name)
        logger.info("Submitting extraction for file: %s", args.pdf)
        result = agent.extract(args.pdf)

        raw_data = normalize_result_data(result)
        save_json(args.raw_output, raw_data)

        extracted = getattr(result, "data", None) or raw_data.get("data") or raw_data
        if hasattr(extracted, "model_dump"):
            extracted = extracted.model_dump()
        elif hasattr(extracted, "dict"):
            extracted = extracted.dict()
        if not isinstance(extracted, dict):
            extracted = {"claims": []}

        save_json(args.output_json, extracted)

        normalized_claims = []
        for item in extracted.get("claims", []):
            if hasattr(item, "model_dump"):
                item = item.model_dump()
            elif hasattr(item, "dict"):
                item = item.dict()
            elif not isinstance(item, dict):
                item = {"claim": str(item)}
            normalized_claims.append({
                "claim": item.get("claim", ""),
                "evidence": item.get("evidence"),
                "page": item.get("page"),
                "confidence": item.get("confidence"),
            })

        save_claims_csv(args.output_csv, normalized_claims)
        logger.info("Done. Extracted %d claims.", len(normalized_claims))
    except Exception as e:
        logger.exception("Extraction failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
