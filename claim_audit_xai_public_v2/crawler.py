from __future__ import annotations

from io import BytesIO
from typing import List, Tuple

import pdfplumber
import requests
from bs4 import BeautifulSoup

try:
    from playwright.sync_api import sync_playwright
except Exception:  # pragma: no cover
    sync_playwright = None

HEADERS = {"User-Agent": "Mozilla/5.0"}


def fetch_govuk_content(url: str) -> Tuple[str, List[str]]:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    content = soup.find("div", class_="govuk-grid-column-two-thirds") or soup.find("main") or soup
    text = content.get_text("\n", strip=True)

    links: List[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "environmental-claims" in href:
            if href.startswith("/"):
                href = "https://www.gov.uk" + href
            links.append(href)

    return text, links


def fetch_pdf(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=40)
    resp.raise_for_status()

    text = []
    with pdfplumber.open(BytesIO(resp.content)) as pdf:
        for page in pdf.pages:
            text.append(page.extract_text() or "")
    return "\n".join(text)


def fetch_dynamic(url: str) -> str:
    if sync_playwright is None:
        return ""

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="networkidle", timeout=30000)
            content = page.content()
            browser.close()
    except Exception:
        return ""

    soup = BeautifulSoup(content, "html.parser")
    return soup.get_text("\n", strip=True)
