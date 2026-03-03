const statusEl = document.getElementById('status');
const metaEl = document.getElementById('meta');
const captureBtn = document.getElementById('captureBtn');
const runBtn = document.getElementById('runBtn');
const downloadBtn = document.getElementById('downloadBtn');
const optionsBtn = document.getElementById('optionsBtn');
const titleInput = document.getElementById('jobTitle');
const companyInput = document.getElementById('company');
const jdInput = document.getElementById('jdText');

let activeRunId = '';
let pollTimer = null;
let capturedSourceUrl = '';

function setStatus(message, kind = '') {
  statusEl.textContent = message;
  statusEl.classList.remove('ok', 'err');
  if (kind) {
    statusEl.classList.add(kind);
  }
}

function setMeta(message) {
  metaEl.textContent = message;
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

function callBackground(type, payload = {}) {
  return new Promise((resolve, reject) => {
    chrome.runtime.sendMessage({ type, payload }, (response) => {
      const runtimeError = chrome.runtime.lastError;
      if (runtimeError) {
        reject(new Error(runtimeError.message));
        return;
      }
      if (!response || !response.ok) {
        reject(new Error(response?.error || 'Unknown extension error.'));
        return;
      }
      resolve(response.result);
    });
  });
}

function setRunReady(ready) {
  downloadBtn.disabled = !ready;
}

async function loadSettingsState() {
  try {
    const settings = await callBackground('settings:get');
    if (!settings.hasApiKey) {
      setStatus('Set API key in options.', 'err');
      runBtn.disabled = true;
      return;
    }
    setStatus('Ready. Capture a job page.', 'ok');
  } catch (error) {
    setStatus(error.message, 'err');
  }
}

async function captureFromActiveTab() {
  setStatus('Capturing job content...');
  setRunReady(false);
  activeRunId = '';
  stopPolling();

  const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
  const activeTab = tabs[0];
  if (!activeTab || !activeTab.id) {
    throw new Error('No active tab detected.');
  }

  const response = await chrome.tabs.sendMessage(activeTab.id, {
    type: 'resumeOptimizer.extractJob',
  });

  if (!response || !response.ok) {
    throw new Error(response?.error || 'Could not extract job content from this page.');
  }

  const payload = response.payload || {};
  const jdText = String(payload.jdText || '').trim();
  if (!jdText) {
    throw new Error('Extractor did not find usable job description text.');
  }

  titleInput.value = String(payload.jobTitle || '').trim();
  companyInput.value = String(payload.company || '').trim();
  jdInput.value = jdText;
  capturedSourceUrl = String(payload.sourceUrl || '').trim();

  const wordCount = jdText.split(/\s+/).filter(Boolean).length;
  setMeta(`Captured ${wordCount} words from ${payload.sourceHost || 'current page'} (${payload.extractor || 'fallback'}).`);
  setStatus('Capture complete. Review then run tailoring.', 'ok');
  runBtn.disabled = false;
}

async function refreshRunStatus() {
  if (!activeRunId) {
    return;
  }
  try {
    const run = await callBackground('api:getRun', { runId: activeRunId });
    const status = String(run.status || 'queued');
    setMeta(`Run ${activeRunId.slice(0, 8)} status: ${status}.`);

    if (status === 'queued' || status === 'running') {
      setStatus(`Tailoring ${status}...`);
      return;
    }

    if (status === 'succeeded') {
      setStatus('Resume ready to download.', 'ok');
      setRunReady(true);
      runBtn.disabled = false;
      stopPolling();
      return;
    }

    const failure = String(run.error || 'Tailoring failed.');
    setStatus(failure, 'err');
    runBtn.disabled = false;
    stopPolling();
  } catch (error) {
    setStatus(error.message, 'err');
    runBtn.disabled = false;
    stopPolling();
  }
}

async function runTailoring() {
  const jdText = String(jdInput.value || '').trim();
  if (!jdText) {
    setStatus('Job description is empty.', 'err');
    return;
  }

  runBtn.disabled = true;
  setRunReady(false);
  stopPolling();

  try {
    const run = await callBackground('api:createRun', {
      jdText,
      jobTitle: String(titleInput.value || '').trim(),
      company: String(companyInput.value || '').trim(),
      sourceUrl: capturedSourceUrl,
    });
    activeRunId = String(run.run_id || run.runId || '').trim();
    if (!activeRunId) {
      throw new Error('No run id returned from API.');
    }
    setStatus('Tailoring queued. Polling status...');
    setMeta(`Run ${activeRunId.slice(0, 8)} created.`);
    pollTimer = setInterval(refreshRunStatus, 2200);
    await refreshRunStatus();
  } catch (error) {
    setStatus(error.message, 'err');
    runBtn.disabled = false;
  }
}

async function downloadResume() {
  if (!activeRunId) {
    setStatus('No completed run available.', 'err');
    return;
  }
  try {
    downloadBtn.disabled = true;
    setStatus('Downloading PDF...');
    const titleSlug = String(titleInput.value || 'tailored-resume')
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-+|-+$/g, '');
    const fileName = `${titleSlug || 'tailored-resume'}.pdf`;
    await callBackground('api:downloadRunPdf', { runId: activeRunId, fileName });
    setStatus('Download started.', 'ok');
  } catch (error) {
    setStatus(error.message, 'err');
  } finally {
    downloadBtn.disabled = false;
  }
}

captureBtn.addEventListener('click', () => {
  captureFromActiveTab().catch((error) => setStatus(error.message, 'err'));
});
runBtn.addEventListener('click', runTailoring);
downloadBtn.addEventListener('click', downloadResume);
optionsBtn.addEventListener('click', () => {
  callBackground('options:open').catch((error) => setStatus(error.message, 'err'));
});

window.addEventListener('beforeunload', stopPolling);
loadSettingsState().catch((error) => setStatus(error.message, 'err'));
