const STORAGE_KEYS = {
  baseUrl: 'resume_optimizer_base_url',
  apiKey: 'resume_optimizer_api_key',
};

const DEFAULT_BASE_URL = 'http://localhost:8030';

function normalizeBaseUrl(rawValue) {
  const raw = String(rawValue || '').trim();
  const fallback = DEFAULT_BASE_URL;
  if (!raw) {
    return fallback;
  }
  return raw.replace(/\/+$/, '');
}

async function getSettings() {
  const stored = await chrome.storage.local.get([STORAGE_KEYS.baseUrl, STORAGE_KEYS.apiKey]);
  return {
    baseUrl: normalizeBaseUrl(stored[STORAGE_KEYS.baseUrl]),
    apiKey: String(stored[STORAGE_KEYS.apiKey] || '').trim(),
  };
}

async function saveSettings(payload) {
  const nextBaseUrl = normalizeBaseUrl(payload?.baseUrl);
  const nextApiKey = String(payload?.apiKey || '').trim();
  await chrome.storage.local.set({
    [STORAGE_KEYS.baseUrl]: nextBaseUrl,
    [STORAGE_KEYS.apiKey]: nextApiKey,
  });
  return {
    baseUrl: nextBaseUrl,
    hasApiKey: Boolean(nextApiKey),
  };
}

function buildHeaders(apiKey, hasBody) {
  const key = String(apiKey || '').trim();
  if (!key) {
    throw new Error('API key is missing. Open extension options and set it first.');
  }
  const headers = {
    Authorization: `Bearer ${key}`,
  };
  if (hasBody) {
    headers['Content-Type'] = 'application/json';
  }
  return headers;
}

async function requestExtensionApi({ baseUrl, apiKey, method, path, body }) {
  const url = `${normalizeBaseUrl(baseUrl)}${path}`;
  const response = await fetch(url, {
    method,
    headers: buildHeaders(apiKey, Boolean(body)),
    body: body ? JSON.stringify(body) : undefined,
  });

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      detail = payload.detail || payload.error || detail;
    } catch (_err) {
      const raw = await response.text();
      if (raw) {
        detail = raw;
      }
    }
    throw new Error(detail);
  }

  const contentType = String(response.headers.get('content-type') || '').toLowerCase();
  if (contentType.includes('application/json')) {
    return response.json();
  }
  return response.arrayBuffer();
}

function assertJobPayload(payload) {
  const jdText = String(payload?.jdText || '').trim();
  if (!jdText) {
    throw new Error('No job description text found. Capture from page before tailoring.');
  }
  return {
    jd_text: jdText,
    source_url: String(payload?.sourceUrl || '').trim(),
    job_title: String(payload?.jobTitle || '').trim(),
    company: String(payload?.company || '').trim(),
  };
}

function arrayBufferToBase64(arrayBuffer) {
  const bytes = new Uint8Array(arrayBuffer);
  const chunkSize = 0x8000;
  let binary = '';
  for (let index = 0; index < bytes.length; index += chunkSize) {
    const chunk = bytes.subarray(index, index + chunkSize);
    binary += String.fromCharCode(...chunk);
  }
  return btoa(binary);
}

function triggerDownload(options) {
  return new Promise((resolve, reject) => {
    chrome.downloads.download(options, (downloadId) => {
      const runtimeError = chrome.runtime.lastError;
      if (runtimeError) {
        reject(new Error(runtimeError.message));
        return;
      }
      resolve(downloadId);
    });
  });
}

async function handleCreateRun(payload) {
  const settings = await getSettings();
  const runPayload = assertJobPayload(payload);
  return requestExtensionApi({
    baseUrl: settings.baseUrl,
    apiKey: settings.apiKey,
    method: 'POST',
    path: '/api/ext/v1/tailor-runs',
    body: runPayload,
  });
}

async function handleGetRun(payload) {
  const runId = String(payload?.runId || '').trim();
  if (!runId) {
    throw new Error('runId is required.');
  }
  const settings = await getSettings();
  return requestExtensionApi({
    baseUrl: settings.baseUrl,
    apiKey: settings.apiKey,
    method: 'GET',
    path: `/api/ext/v1/tailor-runs/${encodeURIComponent(runId)}`,
  });
}

async function handleKeyStatus() {
  const settings = await getSettings();
  return requestExtensionApi({
    baseUrl: settings.baseUrl,
    apiKey: settings.apiKey,
    method: 'GET',
    path: '/api/ext/v1/key/status',
  });
}

async function handleDownloadRunPdf(payload) {
  const runId = String(payload?.runId || '').trim();
  if (!runId) {
    throw new Error('runId is required.');
  }

  const settings = await getSettings();
  const status = await requestExtensionApi({
    baseUrl: settings.baseUrl,
    apiKey: settings.apiKey,
    method: 'GET',
    path: `/api/ext/v1/tailor-runs/${encodeURIComponent(runId)}`,
  });

  if (status.status !== 'succeeded') {
    throw new Error('Run is not complete yet.');
  }
  if (!status.pdf_download_url) {
    throw new Error('Run completed but no PDF URL was returned.');
  }

  const pdfBuffer = await requestExtensionApi({
    baseUrl: settings.baseUrl,
    apiKey: settings.apiKey,
    method: 'GET',
    path: status.pdf_download_url,
  });

  const base64Pdf = arrayBufferToBase64(pdfBuffer);
  const safeFileName = String(payload?.fileName || `resume-${status.job_id || runId}.pdf`).replace(/[^a-zA-Z0-9._-]/g, '_');
  const downloadId = await triggerDownload({
    url: `data:application/pdf;base64,${base64Pdf}`,
    filename: safeFileName,
    saveAs: true,
  });

  return {
    downloadId,
    fileName: safeFileName,
  };
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  (async () => {
    const type = String(message?.type || '');

    if (type === 'settings:get') {
      const settings = await getSettings();
      return {
        baseUrl: settings.baseUrl,
        hasApiKey: Boolean(settings.apiKey),
      };
    }

    if (type === 'settings:save') {
      return saveSettings(message.payload || {});
    }

    if (type === 'api:keyStatus') {
      return handleKeyStatus();
    }

    if (type === 'api:createRun') {
      return handleCreateRun(message.payload || {});
    }

    if (type === 'api:getRun') {
      return handleGetRun(message.payload || {});
    }

    if (type === 'api:downloadRunPdf') {
      return handleDownloadRunPdf(message.payload || {});
    }

    if (type === 'options:open') {
      await chrome.runtime.openOptionsPage();
      return { ok: true };
    }

    throw new Error(`Unsupported action: ${type}`);
  })()
    .then((result) => sendResponse({ ok: true, result }))
    .catch((error) => sendResponse({ ok: false, error: error?.message || String(error) }));

  return true;
});
