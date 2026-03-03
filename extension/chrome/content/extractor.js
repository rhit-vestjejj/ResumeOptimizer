(function () {
  const SITE_SELECTORS = [
    {
      name: 'linkedin',
      selectors: ['.description__text', '.show-more-less-html__markup', '.jobs-description-content__text'],
    },
    {
      name: 'greenhouse',
      selectors: ['#content .content', '.opening', '.job-post'],
    },
    {
      name: 'lever',
      selectors: ['.posting-page .section-wrapper', '.posting-page .content', '.posting-page'],
    },
    {
      name: 'workday',
      selectors: ['[data-automation-id="jobPostingDescription"]', '[data-automation-id="jobPostingMain"]'],
    },
    {
      name: 'indeed',
      selectors: ['#jobDescriptionText', '[data-testid="jobsearch-JobComponent-description"]'],
    },
  ];

  function cleanText(rawText) {
    const text = String(rawText || '')
      .replace(/\r/g, '\n')
      .split('\n')
      .map((line) => line.replace(/\s+/g, ' ').trim())
      .filter(Boolean)
      .join('\n')
      .trim();

    if (text.length > 24000) {
      return text.slice(0, 24000);
    }
    return text;
  }

  function extractLongestTextFromSelectors(selectors) {
    let bestText = '';
    selectors.forEach((selector) => {
      const nodes = Array.from(document.querySelectorAll(selector));
      nodes.forEach((node) => {
        const candidate = cleanText(node.textContent || '');
        if (candidate.length > bestText.length) {
          bestText = candidate;
        }
      });
    });
    return bestText;
  }

  function extractJobTitle() {
    const titleSelectors = ['h1', '[data-testid*="job-title"]', '[class*="job-title"]'];
    for (const selector of titleSelectors) {
      const node = document.querySelector(selector);
      const text = cleanText(node?.textContent || '');
      if (text.length >= 4) {
        return text.slice(0, 140);
      }
    }

    const metaTitle = document.querySelector('meta[property="og:title"]')?.getAttribute('content');
    const fromMeta = cleanText(metaTitle || '');
    if (fromMeta) {
      return fromMeta.slice(0, 140);
    }

    return cleanText(document.title || '').slice(0, 140);
  }

  function extractCompany() {
    const selectors = [
      '[data-testid*="company"]',
      '[class*="company"]',
      '[data-automation-id*="company"]',
      'meta[property="og:site_name"]',
    ];
    for (const selector of selectors) {
      const node = document.querySelector(selector);
      if (!node) {
        continue;
      }
      const content = node.getAttribute('content');
      const text = cleanText(content || node.textContent || '');
      if (text.length >= 2 && text.length < 80) {
        return text;
      }
    }
    return '';
  }

  function extractJobDescription() {
    for (const site of SITE_SELECTORS) {
      const text = extractLongestTextFromSelectors(site.selectors);
      if (text.length >= 500) {
        return { text, extractor: site.name };
      }
    }

    const fallbackSelectors = ['main', 'article', '[role="main"]', 'body'];
    const fallbackText = extractLongestTextFromSelectors(fallbackSelectors);
    return { text: fallbackText, extractor: 'fallback' };
  }

  function extractJobPayload() {
    const jd = extractJobDescription();
    return {
      jdText: jd.text,
      extractor: jd.extractor,
      jobTitle: extractJobTitle(),
      company: extractCompany(),
      sourceUrl: window.location.href,
      sourceHost: window.location.hostname,
    };
  }

  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (!message || message.type !== 'resumeOptimizer.extractJob') {
      return;
    }

    try {
      const payload = extractJobPayload();
      sendResponse({ ok: true, payload });
    } catch (error) {
      sendResponse({ ok: false, error: error?.message || String(error) });
    }
  });
})();
