from __future__ import annotations

import re
from typing import List, Optional

from bs4 import BeautifulSoup
from pydantic import BaseModel, ConfigDict, Field
from readability import Document


class ScrapeResult(BaseModel):
    model_config = ConfigDict(extra='forbid')

    url: str
    title: Optional[str] = None
    company: Optional[str] = None
    jd_text: str
    warnings: List[str] = Field(default_factory=list)


class ScrapeError(RuntimeError):
    pass


def _clean_text(text: str) -> str:
    text = re.sub(r'\r', '\n', text)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    return text.strip()


def _extract_company_from_title(title: Optional[str]) -> Optional[str]:
    if not title:
        return None
    separators = [' at ', ' - ', ' | ', ' — ']
    lowered = title.lower()
    for sep in separators:
        if sep in lowered:
            idx = lowered.index(sep)
            right = title[idx + len(sep):].strip()
            if right and len(right) < 80:
                return right
    return None


def _extract_text_from_html(html: str) -> str:
    doc = Document(html)
    main_html = doc.summary()
    soup = BeautifulSoup(main_html, 'html.parser')
    text = soup.get_text('\n', strip=True)
    if len(text) > 200:
        return _clean_text(text)
    fallback = BeautifulSoup(html, 'html.parser').get_text('\n', strip=True)
    return _clean_text(fallback)


async def scrape_job_posting(url: str, timeout_ms: int = 45000) -> ScrapeResult:
    try:
        from playwright.async_api import async_playwright
    except Exception as exc:
        raise ScrapeError(f'Playwright unavailable: {exc}') from exc

    warnings: List[str] = []
    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True)
            page = await browser.new_page()
            response = await page.goto(url, wait_until='networkidle', timeout=timeout_ms)
            if response is None:
                warnings.append('No HTTP response metadata captured while scraping.')
            html = await page.content()
            title = await page.title()
            await browser.close()
    except Exception as exc:
        raise ScrapeError(f'Failed to scrape URL: {exc}') from exc

    jd_text = _extract_text_from_html(html)
    if len(jd_text) < 200:
        warnings.append('Extracted job description text is short; review and edit manually.')

    return ScrapeResult(
        url=url,
        title=title or None,
        company=_extract_company_from_title(title),
        jd_text=jd_text,
        warnings=warnings,
    )
