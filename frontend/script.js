const API_BASE = (window.REACT_APP_API_BASE || 'http://localhost:8000').replace(/\/$/, '');
const ANNOTATE_PATH = '/annotate';

const endpointLabel = document.getElementById('endpointLabel');
if (endpointLabel) {
  endpointLabel.textContent = API_BASE + ANNOTATE_PATH;
}

const form = document.getElementById('annotateForm');
const textInput = document.getElementById('textInput');
const submitBtn = document.getElementById('submitBtn');
const sampleBtn = document.getElementById('sampleBtn');
const statusEl = document.getElementById('status');
const accuracyEl = document.getElementById('accuracy');
const f1macroEl = document.getElementById('f1macro');
const f1weightedEl = document.getElementById('f1weighted');
const tableBody = document.getElementById('tableBody');

const sampleText = 'The court reviewed the contract to determine whether the assignment of intellectual property rights complied with data protection regulations and cross-border transfer policies.';

if (sampleBtn) {
  sampleBtn.addEventListener('click', () => {
    textInput.value = sampleText;
    textInput.focus();
  });
}

if (form) {
  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const text = textInput.value.trim();
    if (!text) {
      setStatus('Please provide a legal paragraph to annotate.', 'error');
      return;
    }

    setStatus('Submitting text to backend...', 'info');
    toggleLoading(true);

    try {
      const response = await fetch(API_BASE + ANNOTATE_PATH, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });

      if (!response.ok) {
        throw new Error(`Backend responded with ${response.status}`);
      }

      const data = await response.json();
      renderMetrics(extractMetrics(data));
      renderAnnotations(extractAnnotations(data));
      setStatus('Annotation complete.', 'success');
    } catch (err) {
      console.error(err);
      setStatus('Unable to reach the backend or parse the response.', 'error');
    } finally {
      toggleLoading(false);
    }
  });
}

function toggleLoading(isLoading) {
  submitBtn.disabled = isLoading;
  sampleBtn.disabled = isLoading;
  submitBtn.textContent = isLoading ? 'Running...' : 'Run annotation';
}

function setStatus(message, tone = 'info') {
  statusEl.textContent = message;
  statusEl.className = `status${tone === 'error' ? ' error' : tone === 'success' ? ' success' : ''}`;
}

function extractMetrics(data) {
  if (!data) return {};
  if (data.metrics) return data.metrics;
  return {
    accuracy: data.accuracy,
    f1_macro: data.f1_macro,
    f1_weighted: data.f1_weighted
  };
}

function renderMetrics(metrics = {}) {
  accuracyEl.textContent = formatNumber(metrics.accuracy);
  f1macroEl.textContent = formatNumber(metrics.f1_macro);
  f1weightedEl.textContent = formatNumber(metrics.f1_weighted);
}

function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return '--';
  if (typeof value === 'string' && value.trim() === '') return '--';
  const num = Number(value);
  return Number.isFinite(num) ? num.toFixed(3) : '--';
}

function extractAnnotations(data) {
  if (!data) return [];
  if (Array.isArray(data.annotations)) return data.annotations;
  if (Array.isArray(data.tokens)) return data.tokens;
  if (Array.isArray(data.predictions)) return data.predictions;
  return [];
}

function renderAnnotations(items) {
  tableBody.innerHTML = '';
  if (!items.length) {
    tableBody.innerHTML = '<tr><td colspan="4">No annotations returned. Ensure the backend responds with a tokens/annotations array.</td></tr>';
    return;
  }

  items.forEach((item) => {
    const row = document.createElement('tr');
    const token = escapeHtml(item.token || item.word || item.text || '');
    const sense = escapeHtml(item.synset || item.label || item.prediction || '--');
    const definition = escapeHtml(item.definition || item.gloss || '');
    const confidence = item.confidence !== undefined ? formatNumber(item.confidence) : '--';

    row.innerHTML = `
      <td>${token}</td>
      <td>${sense}</td>
      <td>${definition || '<span style="color:#7b8798;">(no definition provided)</span>'}</td>
      <td>${confidence}</td>
    `;
    tableBody.appendChild(row);
  });
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
