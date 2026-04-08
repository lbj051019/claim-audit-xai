from __future__ import annotations
from typing import Dict, List

from openai import OpenAI

from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)


def verify_evidence(claim: str, evidence: str) -> str:
    prompt = f"""
You are a compliance analyst.

Claim:
{claim}

Evidence:
{evidence}

Does the evidence support the claim?

Answer ONLY:
supported / partially supported / not supported
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )

    result = response.choices[0].message.content.lower().strip()
    return result.replace("-", "").strip()


def extract_features(claim: str, evidence: str) -> Dict[str, int]:
    return {
        "has_evidence": int(len(evidence) > 50),
        "has_numbers": int(any(char.isdigit() for char in claim)),
        "has_absolute_terms": int(any(w in claim.lower() for w in ["zero", "always", "never"])),
        "has_uncertainty": int(any(w in claim.lower() for w in ["aim", "may", "could", "plan"])),
        "third_party": int("independent" in evidence.lower() or "verified" in evidence.lower()),
    }


def score_claim(features: Dict[str, int]) -> float:
    score = 0.0
    if features["has_evidence"]:
        score += 0.3
    if features["third_party"]:
        score += 0.3
    if features["has_numbers"]:
        score += 0.1
    if features["has_absolute_terms"]:
        score -= 0.4
    if features["has_uncertainty"]:
        score -= 0.1
    return max(0.0, min(1.0, score))


def verdict(score: float) -> str:
    if score > 0.6:
        return "Supported"
    if score > 0.3:
        return "Partially supported"
    return "Unsubstantiated"


def explain_shap(features: Dict[str, int], support_level: str) -> List[str]:
    explanations: List[str] = []

    if features["has_evidence"]:
        explanations.append("Evidence found")
    else:
        explanations.append("No supporting evidence")

    if features["third_party"]:
        explanations.append("Third-party validation increases trust")

    if features["has_numbers"]:
        explanations.append("Quantitative data in claim")

    if features["has_absolute_terms"]:
        explanations.append("Absolute wording increases risk")

    if features["has_uncertainty"]:
        explanations.append("Uncertain wording weakens claim")

    explanations.append(f"Evidence validation: {support_level}")
    return explanations


def evaluate_claims(claims: List[str], rag) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []

    for claim in claims:
        evidence_chunks = rag.search(claim)
        evidence = evidence_chunks[0] if evidence_chunks else ""

        support_level = verify_evidence(claim, evidence)
        features = extract_features(claim, evidence)
        score = score_claim(features)

        if support_level == "supported":
            score += 0.2
        elif support_level == "partially supported":
            score += 0.1
        else:
            score -= 0.2

        score = max(0.0, min(1.0, score))
        final_verdict = verdict(score)
        explanation = explain_shap(features, support_level)

        results.append({
            "claim": claim,
            "evidence": evidence[:300],
            "support_level": support_level,
            "verdict": final_verdict,
            "score": round(score, 2),
            "explanation": explanation,
        })

    return results
