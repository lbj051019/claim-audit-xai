# 🧠 ESG Claim Audit Systems (XAI + RAG)

An AI-powered system for auditing environmental and sustainability claims (greenwashing detection) using explainable AI and regulatory guidance.

\---

## 🚀 Overview

This repository contains **two versions** of an ESG claim auditing system:

### 🔹 v1 – Rule-based + XAI (SHAP)

* Feature engineering (absolute claims, quantified claims, etc.)
* Random Forest scoring
* SHAP-based explainability
* Regulatory alignment (CMA, MMO, OHI)

### 🔹 v2 – RAG + LLM + XAI

* Web scraping (CMA, MMO, OHI sources)
* Retrieval-Augmented Generation (FAISS)
* LLM-based evidence verification
* Rule-based scoring + explainability
* End-to-end automated pipeline

\---

## 🏗️ Project Structure

```
.
├── claim\_audit\_xai\_public\_v1/
├── claim\_audit\_xai\_public\_v2/
```

\---

## 🧠 v1 – Rule-based XAI System

Run:

```
python claim\_audit\_xai.py --claims examples/demo\_claims.txt
```

\---

## 🤖 v2 – RAG + LLM System

Run:

```
python full\_pipeline.py
```

\---

## 🔑 Requirements

* Python 3.10+
* OpenAI API key
* Llama Cloud API key

\---

## ⚠️ Notes

* Research prototype
* Data is anonymised
* Not legal advice

\---

## 👤 Author

Bingji Li

