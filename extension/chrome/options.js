const baseUrlInput = document.getElementById('baseUrl');
const apiKeyInput = document.getElementById('apiKey');
const saveBtn = document.getElementById('saveBtn');
const testBtn = document.getElementById('testBtn');
const toggleKeyBtn = document.getElementById('toggleKeyBtn');
const statusEl = document.getElementById('status');

let showKey = false;

function setStatus(message, kind = '') {
  statusEl.textContent = message;
  statusEl.classList.remove('ok', 'err');
  if (kind) {
    statusEl.classList.add(kind);
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

async function loadSettings() {
  const settings = await callBackground('settings:get');
  baseUrlInput.value = settings.baseUrl || 'http://localhost:8030';
}

async function saveSettings() {
  setStatus('Saving settings...');
  const result = await callBackground('settings:save', {
    baseUrl: baseUrlInput.value,
    apiKey: apiKeyInput.value,
  });
  setStatus(`Saved. API key ${result.hasApiKey ? 'configured' : 'missing'}.`, 'ok');
}

async function testKey() {
  setStatus('Testing API key...');
  const status = await callBackground('api:keyStatus');
  if (!status.has_key) {
    setStatus('Key is accepted but backend reports no active key.', 'err');
    return;
  }
  setStatus(`Key valid (id: ${status.key_id}).`, 'ok');
}

function toggleKeyVisibility() {
  showKey = !showKey;
  apiKeyInput.type = showKey ? 'text' : 'password';
  toggleKeyBtn.textContent = showKey ? 'Hide Key' : 'Show Key';
}

saveBtn.addEventListener('click', () => {
  saveSettings().catch((error) => setStatus(error.message, 'err'));
});

testBtn.addEventListener('click', () => {
  testKey().catch((error) => setStatus(error.message, 'err'));
});

toggleKeyBtn.addEventListener('click', toggleKeyVisibility);

loadSettings()
  .then(() => setStatus('Loaded settings.'))
  .catch((error) => setStatus(error.message, 'err'));
