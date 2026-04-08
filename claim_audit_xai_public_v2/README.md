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

## Repository structure

- `extract_claims_sdk.py` - claim extraction from a PDF using LlamaExtract
- `claim_filter.py` - filters environmental / sustainability claims with OpenAI
- `main.py` - RAG retrieval, scoring, explanation, and report generation
- `claims_pipeline.py` - evidence verification and feature-based scoring
- `rag.py` - FAISS retrieval over official guidance sources
- `crawler.py` - HTML / PDF / dynamic page collection
- `processor.py` - text cleaning and chunking
- `report_generator.py` - writes TXT and JSON reports
- `run_pipeline.py` - one-command runner for the full workflow

## Quick start

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
export LLAMA_CLOUD_API_KEY=llx-...
python run_pipeline.py --pdf your_report.pdf
```

## Demo mode

You can also run the final analysis stage with the included synthetic sample claims:

```bash
python main.py --claims-file examples/demo_claims.txt
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
- The example claims in `examples/demo_claims.txt` are synthetic and safe to publish.
