# ClaimAuditXAI

An end-to-end, explainable AI pipeline for extracting, analysing, and auditing environmental and sustainability claims from documents.

## What it does

- Extracts claims from PDFs
- Identifies environmental or sustainability claims
- Evaluates them against official guidance
- Produces a structured score and explanation
- Uses SHAP to show which features drove the result

## Pipeline

```text
PDF -> Claim Extraction -> Environmental Claim Filtering -> Audit + Scoring + SHAP -> Report
```

## Files

- `extract_claims_sdk.py` — claim extraction from a PDF using LlamaExtract
- `claim_filter.py` — filters environmental / sustainability claims with OpenAI
- `claim_audit_xai.py` — audits claims with rule-based scoring and SHAP explanations
- `run_pipeline.py` — optional command-line helper to run all steps

## Quick start

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
export LLAMA_CLOUD_API_KEY=llx-...
python run_pipeline.py --pdf your_report.pdf
```

## Outputs

- `extracted_claims.json`
- `extracted_claims.csv`
- `environmental_claims.txt`
- `report.txt`
- `report.json`

## Notes

- This is a research prototype.
- The scoring layer is heuristic and meant for analysis, not legal advice.
- The example files in `examples/` are synthetic and safe to publish.
