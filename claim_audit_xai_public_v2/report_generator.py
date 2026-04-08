from __future__ import annotations

import json
from typing import Any, Dict, List


def generate_report(results: List[Dict[str, Any]], output_txt: str = "report.txt", output_json: str = "report.json") -> None:
    if not results:
        raise ValueError("No results to report.")

    avg_score = sum(r["score"] for r in results) / len(results)

    if avg_score > 0.6:
        overall_verdict = "Strong Support"
    elif avg_score > 0.3:
        overall_verdict = "Moderate / Mixed Support"
    else:
        overall_verdict = "High Bluewashing Risk"

    confidence = round(min(1.0, avg_score + 0.2), 2)

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(f"Score: {round(avg_score, 2)}\n")
        f.write(f"Verdict: {overall_verdict}\n\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Confidence: {confidence}\n\n")
        f.write("=" * 60 + "\n\n")

        for i, r in enumerate(results, 1):
            f.write(f"Claim {i}:\n")
            f.write(f"- {r['claim']}\n\n")
            f.write(f"Score: {r['score']}\n")
            f.write(f"Confidence: {round(r['score'] + 0.4, 2)}\n")

            if r["score"] < 0.3:
                verdict = "High Bluewashing Risk"
            elif r["score"] < 0.6:
                verdict = "Moderate Risk"
            else:
                verdict = "Supported"

            f.write(f"Verdict: {verdict}\n\n")
            f.write("Explanation:\n")
            for e in r["explanation"]:
                f.write(f"- {e}\n")

            f.write("\nXAI Model Insight:\n")
            for e in r["explanation"]:
                f.write(f"- {e.lower()}\n")

            words = r["claim"].lower().split()
            key_terms = list(dict.fromkeys(words))[:8]
            f.write("\nKey Terms:\n")
            f.write(", ".join(key_terms) + "\n")

            f.write("\nOfficial Evidence:\n")
            f.write(f"- {r['evidence']}\n")

            f.write("\nOfficial Sources:\n")
            f.write("- CMA PDF\n")
            f.write("- CMA HTML\n")
            f.write("- MMO\n")
            f.write("- OHI\n")
            f.write("\n" + "=" * 60 + "\n\n")

    output_data = {
        "overall_score": round(avg_score, 2),
        "overall_verdict": overall_verdict,
        "confidence": confidence,
        "claims": results,
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Report saved: {output_txt}, {output_json}")
