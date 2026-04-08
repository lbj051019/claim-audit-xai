from __future__ import annotations
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

URLS = {
    "CMA_PDF": "https://assets.publishing.service.gov.uk/media/61482fd4e90e070433f6c3ea/Guidance_for_businesses_on_making_environmental_claims_.pdf",
    "CMA_HTML": "https://www.gov.uk/government/publications/green-claims-code-making-environmental-claims",
    "MMO": "https://www.gov.uk/government/organisations/marine-management-organisation",
    "OHI": "https://oceanhealthindex.org/resources/data/",
}
