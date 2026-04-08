#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Audit environmental claims against official guidance and explain the result with SHAP."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass, asdict
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import shap
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text
from sklearn.ensemble import RandomForestRegressor

SOURCE_CONFIG = {
    "cma_pdf": "https://assets.publishing.service.gov.uk/media/61482fd4e90e070433f6c3ea/Guidance_for_businesses_on_making_environmental_claims_.pdf",
    "cma_html": "https://www.gov.uk/government/publications/green-claims-code-making-environmental-claims",
    "mmo": "https://www.gov.uk/government/organisations/marine-management-organisation",
    "ohi": "https://oceanhealthindex.org/resources/data/",
}
HEADERS = {"User-Agent": "ClaimAuditXAI/1.0"}
STOPWORDS = {"the", "a", "an", "of", "to", "and", "or", "for", "in", "on", "by", "with", "is", "are", "be", "as", "at", "that", "this", "it", "from", "their", "they", "can", "may", "more", "most", "than", "up", "into", "under", "over", "one", "using", "use", "used", "based", "such"}
ABSOLUTE_WORDS = ["zero harm", "zero impact", "no harm", "no impact", "always", "never", "guarantee", "completely safe", "ocean-positive"]
FEATURE_LABELS = {
    "has_absolute_claim": "the claim uses absolute wording",
    "is_quantified": "the claim contains a quantified/comparative statement",
    "theme_carbon": "the claim is about carbon sequestration/storage",
    "theme_biodiversity": "the claim is about biodiversity/ecosystem outcomes",
    "theme_agriculture": "the claim is about agriculture/soil/crop effects",
    "theme_marine": "the claim is about marine harvesting/coastal/aquaculture activities",
    "has_cma_evidence": "CMA guidance requires evidence for this type of claim",
    "has_cma_quantify": "CMA guidance expects quantification/baseline context",
    "has_cma_absolute_warning": "CMA guidance warns against broad/absolute wording",
    "has_cma_third_party": "CMA guidance refers to independent/third-party verification",
    "has_mmo_licence": "MMO guidance indicates licensing requirements may apply",
    "has_mmo_eia": "MMO guidance indicates EIA/environmental review may apply",
    "has_mmo_monitoring": "MMO guidance indicates monitoring requirements may apply",
    "has_ohi_carbon": "OHI provides carbon-related baseline concepts/data",
    "has_ohi_biodiversity": "OHI provides biodiversity/coastal baseline concepts/data",
    "has_ohi_data": "OHI provides supporting data resources",
    "num_cma_evidence": "multiple relevant CMA passages were found",
    "num_mmo_evidence": "multiple relevant MMO passages were found",
    "num_ohi_evidence": "multiple relevant OHI passages were found",
}
CMA_CATEGORIES = {"evidence": ["supported by evidence", "substantiated", "evidence", "misleading", "must not mislead"], "quantify": ["baseline", "comparator", "life cycle", "lifecycle", "quantif"], "absolute": ["absolute claims", "avoid words such as", "zero", "broad and absolute claims"], "third_party": ["third party", "independent", "independently verified"]}
MMO_CATEGORIES = {"licence": ["marine licence", "marine license", "licensing", "licence"], "eia": ["environmental impact assessment", "eia"], "monitoring": ["monitoring", "monitor"]}
OHI_CATEGORIES = {"carbon": ["carbon", "storage"], "biodiversity": ["biodiversity", "coastal protection", "habitat"], "data": ["data", "download", "github", "scores"]}


@dataclass
class EvidenceSnippet:
    source: str
    category: str
    text: str
    score: float


@dataclass
class ClaimResult:
    claim: str
    support_score: float
    verdict: str
    confidence: float
    explanation: str
    xai_insight: List[str]
    key_terms: List[str]
    official_evidence: List[Dict[str, Any]]
    features: Dict[str, float]


def fetch_html_text(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return "\n".join(t.strip() for t in soup.stripped_strings if t.strip())


def fetch_pdf_text(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=40)
    r.raise_for_status()
    return extract_text(BytesIO(r.content))


def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{2,}", "\n", text).strip()


def split_into_passages(text: str, min_len: int = 40) -> List[str]:
    return [clean_text(c) for c in re.split(r"\n+", text) if len(clean_text(c)) >= min_len]


def load_official_sources() -> Dict[str, Dict[str, Any]]:
    cma_pdf = clean_text(fetch_pdf_text(SOURCE_CONFIG["cma_pdf"]))
    cma_html = clean_text(fetch_html_text(SOURCE_CONFIG["cma_html"]))
    mmo = clean_text(fetch_html_text(SOURCE_CONFIG["mmo"]))
    ohi = clean_text(fetch_html_text(SOURCE_CONFIG["ohi"]))
    return {
        "cma": {"url_pdf": SOURCE_CONFIG["cma_pdf"], "url_html": SOURCE_CONFIG["cma_html"], "text": cma_pdf + "\n" + cma_html, "passages": split_into_passages(cma_pdf + "\n" + cma_html)},
        "mmo": {"url": SOURCE_CONFIG["mmo"], "text": mmo, "passages": split_into_passages(mmo)},
        "ohi": {"url": SOURCE_CONFIG["ohi"], "text": ohi, "passages": split_into_passages(ohi)},
    }


def normalize_tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-zA-Z0-9%\-]+", text.lower()) if t not in STOPWORDS and len(t) > 1]


def extract_key_terms(claim: str) -> List[str]:
    out: List[str] = []
    for t in normalize_tokens(claim):
        if t not in out:
            out.append(t)
    return out[:8]


def claim_themes(claim: str) -> Dict[str, int]:
    c = claim.lower()
    return {
        "theme_carbon": int(any(k in c for k in ["carbon", "co2", "sequestr", "biochar", "sink"])),
        "theme_biodiversity": int(any(k in c for k in ["biodiversity", "ecosystem", "species", "habitat", "restore", "restoration"])),
        "theme_agriculture": int(any(k in c for k in ["soil", "crop", "yield", "fertilizer", "fertiliser", "biostimulant"])),
        "theme_marine": int(any(k in c for k in ["seaweed", "marine", "coastal", "harvest", "harvesting", "aquaculture", "fishmeal", "ocean", "kelp"])),
    }


def has_absolute_claim(claim: str) -> int:
    c = claim.lower()
    return int(any(w in c for w in ABSOLUTE_WORDS))


def is_quantified(claim: str) -> int:
    c = claim.lower()
    return int(bool(re.search(r"(\bup to\b|\d+%|\bpercent\b|%|\b\d+\s*times\b|\bfaster\b|\bless\b|\bmore\b)", c)))


def choose_categories_for_claim(claim: str) -> Dict[str, List[str]]:
    c = claim.lower()
    chosen = {"cma": ["evidence"], "mmo": [], "ohi": []}
    if is_quantified(claim):
        chosen["cma"].append("quantify")
    if has_absolute_claim(claim) or any(k in c for k in ["most efficient", "best", "nature's most", "always", "never"]):
        chosen["cma"].append("absolute")
    chosen["cma"].append("third_party")
    if any(k in c for k in ["seaweed", "marine", "coastal", "harvest", "harvesting", "aquaculture", "fishmeal", "kelp"]):
        chosen["mmo"].extend(["licence", "eia", "monitoring"])
    if any(k in c for k in ["carbon", "co2", "sink", "sequestr", "biochar"]):
        chosen["ohi"].extend(["carbon", "data"])
    if any(k in c for k in ["biodiversity", "ecosystem", "species", "habitat", "restore", "restoration", "coastal"]):
        chosen["ohi"].extend(["biodiversity", "data"])
    return {k: list(dict.fromkeys(v)) for k, v in chosen.items()}


def overlap_score(claim: str, passage: str, extra_terms: List[str]) -> float:
    claim_tokens = set(normalize_tokens(claim)); passage_tokens = set(normalize_tokens(passage))
    if not passage_tokens:
        return 0.0
    jaccard = len(claim_tokens & passage_tokens) / max(1, len(claim_tokens | passage_tokens))
    extra_hit = sum(1 for t in extra_terms if t in passage.lower())
    return jaccard * 5.0 + extra_hit * 0.7


def retrieve_relevant_evidence(claim: str, official_sources: Dict[str, Dict[str, Any]], max_per_source: int = 2) -> List[EvidenceSnippet]:
    chosen = choose_categories_for_claim(claim)
    all_snippets: List[EvidenceSnippet] = []
    for cat in chosen["cma"]:
        for p in official_sources["cma"]["passages"]:
            score = overlap_score(claim, p, CMA_CATEGORIES.get(cat, []))
            if score > 0.7:
                all_snippets.append(EvidenceSnippet("CMA", cat, p, score))
    for cat in chosen["mmo"]:
        for p in official_sources["mmo"]["passages"]:
            score = overlap_score(claim, p, MMO_CATEGORIES.get(cat, []))
            if score > 0.7:
                all_snippets.append(EvidenceSnippet("MMO", cat, p, score))
    for cat in chosen["ohi"]:
        for p in official_sources["ohi"]["passages"]:
            score = overlap_score(claim, p, OHI_CATEGORIES.get(cat, []))
            if score > 0.7:
                all_snippets.append(EvidenceSnippet("OHI", cat, p, score))
    dedup = {}
    for s in all_snippets:
        key = (s.source, s.text)
        if key not in dedup or s.score > dedup[key].score:
            dedup[key] = s
    snippets = sorted(dedup.values(), key=lambda x: x.score, reverse=True)
    final: List[EvidenceSnippet] = []
    counter = Counter()
    for s in snippets:
        if counter[s.source] < max_per_source:
            final.append(s)
            counter[s.source] += 1
    return final


def build_feature_vector(claim: str, evidence: List[EvidenceSnippet]) -> Dict[str, float]:
    feats = {"has_absolute_claim": float(has_absolute_claim(claim)), "is_quantified": float(is_quantified(claim))}
    feats.update({k: float(v) for k, v in claim_themes(claim).items()})
    cma_texts = [e.text.lower() for e in evidence if e.source == "CMA"]
    mmo_texts = [e.text.lower() for e in evidence if e.source == "MMO"]
    ohi_texts = [e.text.lower() for e in evidence if e.source == "OHI"]
    feats["has_cma_evidence"] = float(any(any(t in p for t in ["evidence", "substantiated", "mislead"]) for p in cma_texts))
    feats["has_cma_quantify"] = float(any(any(t in p for t in ["baseline", "comparator", "life cycle", "lifecycle", "quantif"]) for p in cma_texts))
    feats["has_cma_absolute_warning"] = float(any(any(t in p for t in ["absolute claims", "avoid words such as", "zero", "broad and absolute claims"]) for p in cma_texts))
    feats["has_cma_third_party"] = float(any(any(t in p for t in ["third party", "independent", "independently verified"]) for p in cma_texts))
    feats["has_mmo_licence"] = float(any(any(t in p for t in ["marine licence", "marine license", "licence", "licensing"]) for p in mmo_texts))
    feats["has_mmo_eia"] = float(any(any(t in p for t in ["environmental impact assessment", "eia"]) for p in mmo_texts))
    feats["has_mmo_monitoring"] = float(any(any(t in p for t in ["monitoring", "monitor"]) for p in mmo_texts))
    feats["has_ohi_carbon"] = float(any("carbon" in p for p in ohi_texts))
    feats["has_ohi_biodiversity"] = float(any(any(t in p for t in ["biodiversity", "coastal protection", "habitat"]) for p in ohi_texts))
    feats["has_ohi_data"] = float(any(any(t in p for t in ["data", "download", "scores", "github"]) for p in ohi_texts))
    feats["num_cma_evidence"] = float(min(sum(1 for e in evidence if e.source == "CMA"), 3))
    feats["num_mmo_evidence"] = float(min(sum(1 for e in evidence if e.source == "MMO"), 3))
    feats["num_ohi_evidence"] = float(min(sum(1 for e in evidence if e.source == "OHI"), 3))
    return feats


def rule_support_score(feats: Dict[str, float], evidence: List[EvidenceSnippet]) -> Tuple[float, str]:
    score = 0.55
    if feats["has_absolute_claim"]:
        score -= 0.5
    if feats["is_quantified"] and not feats["has_cma_quantify"]:
        score -= 0.25
    if feats["theme_marine"]:
        score -= 0.10
    if feats["has_cma_evidence"]:
        score += 0.08
    if feats["has_ohi_carbon"] or feats["has_ohi_biodiversity"]:
        score += 0.08
    if feats["num_cma_evidence"] >= 2:
        score += 0.05
    if len(evidence) == 0:
        score -= 0.20
    score = max(0.0, min(1.0, score))
    verdict = "Severe Bluewashing" if score < 0.20 else "High Bluewashing Risk" if score < 0.40 else "Partially Supported" if score < 0.65 else "Supported"
    return score, verdict


def build_explanation_text(verdict: str, feats: Dict[str, float], evidence: List[EvidenceSnippet]) -> str:
    parts: List[str] = []
    if verdict in {"Severe Bluewashing", "High Bluewashing Risk"}:
        if feats["has_absolute_claim"] and feats["has_cma_absolute_warning"]:
            parts.append("The claim uses broad or absolute wording that CMA guidance treats as high-risk unless it is narrowly and clearly substantiated.")
        if feats["is_quantified"] and not feats["has_cma_quantify"]:
            parts.append("It also makes a quantified or comparative statement without direct official support for the comparison framework or baseline.")
        if len(evidence) == 0:
            parts.append("No direct official support was found in the claim-specific retrieval.")
    elif verdict == "Partially Supported":
        parts.append("Some official material is relevant to the topic, but it does not directly substantiate the full wording of the claim.")
        if feats["is_quantified"]:
            parts.append("The quantified or comparative wording appears to require stronger evidence or clearer boundaries.")
    else:
        parts.append("Relevant official material was found and the wording is closer to what can be supported, although it should still be carefully bounded in public use.")
    if feats["theme_marine"] and (feats["has_mmo_licence"] or feats["has_mmo_eia"] or feats["has_mmo_monitoring"]):
        parts.append("Because the claim involves marine activity, MMO-related licensing, EIA, or monitoring expectations are relevant.")
    if feats["theme_carbon"] and feats["has_ohi_carbon"]:
        parts.append("OHI material provides carbon-related baseline concepts, but that is not equivalent to direct proof of the specific claim.")
    return " ".join(parts).strip()


def train_xai_model(feature_rows: List[Dict[str, float]], target_scores: List[float]):
    df = pd.DataFrame(feature_rows)
    X = df.values
    y = np.array(target_scores, dtype=float)
    model = RandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=1)
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    tree_preds = np.array([tree.predict(X) for tree in model.estimators_])
    confidence = 1.0 - np.clip(tree_preds.std(axis=0) / 0.25, 0, 1)
    return df.columns.tolist(), shap_values, confidence


def build_xai_lines(feature_names: List[str], shap_row: np.ndarray, feature_row: Dict[str, float]) -> List[str]:
    items = [(f, float(shap_row[i]), float(feature_row[f])) for i, f in enumerate(feature_names)]
    items.sort(key=lambda x: abs(x[1]), reverse=True)
    lines: List[str] = []
    for f, contrib, _ in items[:4]:
        if abs(contrib) < 0.005:
            continue
        label = FEATURE_LABELS.get(f, f)
        direction = "increased" if contrib > 0 else "decreased"
        lines.append(f"{label}, which {direction} support for this claim")
    return lines or ["the model did not identify any strong feature-level signals"]


def render_report(results: List[ClaimResult], official_sources: Dict[str, Dict[str, Any]]) -> str:
    scores = [r.support_score for r in results]
    overall_score = float(np.mean(scores)) if scores else 0.0
    overall_conf = float(np.mean([r.confidence for r in results])) if results else 0.0
    if overall_score < 0.20:
        overall_verdict = "Severe Bluewashing"
    elif overall_score < 0.40:
        overall_verdict = "High Bluewashing Risk"
    elif overall_score < 0.65:
        overall_verdict = "Moderate / Mixed Support"
    else:
        overall_verdict = "Mostly Supported"
    lines = [f"Score: {overall_score:.2f}", f"Verdict: {overall_verdict}", "", "=" * 60, "", f"Confidence: {overall_conf:.2f}", "", "=" * 60, ""]
    for idx, r in enumerate(results, start=1):
        lines += ["=" * 60, f"Claim {idx}:", r.claim, "", f"Score: {r.support_score:.2f}", f"Confidence: {r.confidence:.2f}", f"Verdict: {r.verdict}", "", "Explanation:", r.explanation, "", "XAI Model Insight:"]
        lines += [f"- {x}" for x in r.xai_insight]
        lines += ["", "Key Terms:", ", ".join(r.key_terms) if r.key_terms else "(none)", "", "Official Evidence:"]
        lines += [f"- [{ev['source']} / {ev['category']}] {ev['text']}" for ev in r.official_evidence] or ["- No direct official support found in claim-specific retrieval."]
        lines += ["", "Official Sources:", f"- CMA PDF: {official_sources['cma']['url_pdf']}", f"- CMA HTML: {official_sources['cma']['url_html']}", f"- MMO: {official_sources['mmo']['url']}", f"- OHI: {official_sources['ohi']['url']}", "", "=" * 60, ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claims", required=True, help="Path to claims txt file, one claim per line")
    parser.add_argument("--report-txt", default="report.txt")
    parser.add_argument("--report-json", default="report.json")
    args = parser.parse_args()

    path = Path(args.claims)
    if not path.exists():
        raise FileNotFoundError(f"Claims file not found: {path}")
    claims = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not claims:
        raise ValueError("No claims found in txt file.")

    print(f"Loaded {len(claims)} claims.")
    print("Loading official sources...")
    official_sources = load_official_sources()

    rows: List[Dict[str, float]] = []
    scores: List[float] = []
    evidences: List[List[EvidenceSnippet]] = []
    verdicts: List[str] = []
    explanations: List[str] = []
    terms: List[List[str]] = []

    print("Scoring claims...")
    for i, claim in enumerate(claims, start=1):
        ev = retrieve_relevant_evidence(claim, official_sources)
        feats = build_feature_vector(claim, ev)
        score, verdict = rule_support_score(feats, ev)
        expl = build_explanation_text(verdict, feats, ev)
        rows.append(feats); scores.append(score); evidences.append(ev); verdicts.append(verdict); explanations.append(expl); terms.append(extract_key_terms(claim))
        print(f"[{i}/{len(claims)}] score={score:.2f} verdict={verdict}")

    print("Running SHAP explanation layer...")
    feature_names, shap_values, confidence = train_xai_model(rows, scores)
    results: List[ClaimResult] = []
    for i, claim in enumerate(claims):
        ev_out = [{"source": e.source, "category": e.category, "text": e.text[:500] + ("..." if len(e.text) > 500 else ""), "relevance_score": round(e.score, 3)} for e in evidences[i]]
        results.append(ClaimResult(claim=claim, support_score=round(scores[i], 2), verdict=verdicts[i], confidence=round(float(confidence[i]), 2), explanation=explanations[i], xai_insight=build_xai_lines(feature_names, shap_values[i], rows[i]), key_terms=terms[i], official_evidence=ev_out, features={k: round(float(v), 3) for k, v in rows[i].items()}))

    Path(args.report_txt).write_text(render_report(results, official_sources), encoding="utf-8")
    Path(args.report_json).write_text(json.dumps({"official_sources": {"cma_pdf": official_sources["cma"]["url_pdf"], "cma_html": official_sources["cma"]["url_html"], "mmo": official_sources["mmo"]["url"], "ohi": official_sources["ohi"]["url"]}, "num_claims": len(results), "results": [asdict(r) for r in results]}, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nSummary:")
    for idx, r in enumerate(results, start=1):
        print(f"{idx}. score={r.support_score:.2f} | confidence={r.confidence:.2f} | verdict={r.verdict}")
    print(f"\nSaved full report: {args.report_txt}")
    print(f"Saved structured report: {args.report_json}")


if __name__ == "__main__":
    main()
