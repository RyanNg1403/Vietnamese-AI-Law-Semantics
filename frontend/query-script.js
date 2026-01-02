const API_BASE = (window.REACT_APP_API_BASE || 'http://localhost:3030').replace(/\/$/, '');

let queries = [];
let currentResponse = null;

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
  await loadQueries();
  setupEventListeners();
});

// Load predefined queries
async function loadQueries() {
  const statusEl = document.getElementById('queryLoadStatus');

  try {
    if (statusEl) statusEl.textContent = 'Loading queries...';
    console.log('Fetching queries from:', `${API_BASE}/queries`);
    const response = await fetch(`${API_BASE}/queries`);

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    console.log('Queries loaded:', data);
    queries = data.queries;

    const select = document.getElementById('querySelect');
    if (!select) {
      console.error('querySelect element not found!');
      if (statusEl) statusEl.textContent = 'Error: UI element not found';
      return;
    }

    queries.forEach(q => {
      const option = document.createElement('option');
      option.value = q.id;
      option.textContent = `${q.id}. ${q.nl}`;
      select.appendChild(option);
    });

    console.log(`Loaded ${queries.length} queries successfully`);
    if (statusEl) {
      statusEl.textContent = `✓ Loaded ${queries.length} queries`;
      statusEl.style.color = '#4ade80';
    }

    select.addEventListener('change', (e) => {
      const queryId = parseInt(e.target.value);
      const query = queries.find(q => q.id === queryId);
      if (query) {
        document.getElementById('queryText').textContent = query.query;
        document.getElementById('queryNL').textContent = query.nl;
        document.getElementById('queryInfo').style.display = 'block';
        document.getElementById('runQueryBtn').disabled = false;
      } else {
        document.getElementById('queryInfo').style.display = 'none';
        document.getElementById('runQueryBtn').disabled = true;
      }
    });
  } catch (error) {
    console.error('Failed to load queries:', error);
    if (statusEl) {
      statusEl.textContent = `✗ Failed to load queries: ${error.message}`;
      statusEl.style.color = '#ef4444';
    }
    alert(`Failed to load queries from backend at ${API_BASE}/queries\n\nError: ${error.message}\n\nPlease ensure the backend is running on port 3030.`);
  }
}

// Setup event listeners
function setupEventListeners() {
  document.getElementById('runQueryBtn').addEventListener('click', runQuery);
}

// Run query
async function runQuery() {
  const queryId = parseInt(document.getElementById('querySelect').value);
  const wsdMethod = document.querySelector('input[name="wsdMethod"]:checked').value;

  if (!queryId) return;

  // Show pipeline
  document.getElementById('pipelineSection').style.display = 'block';
  document.getElementById('detailsSection').style.display = 'block';
  document.getElementById('resultSection').style.display = 'block';

  // Reset pipeline
  resetPipeline();

  // Update status
  const statusEl = document.getElementById('pipelineStatus');
  statusEl.textContent = 'Processing...';
  statusEl.className = 'status processing';

  try {
    // Make API call
    const response = await fetch(`${API_BASE}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query_id: queryId, wsd_method: wsdMethod })
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    currentResponse = data;

    // Animate pipeline
    await animatePipeline(data);

    // Render details
    renderWSDDetails(data.steps.wsd);
    renderFOLDetails(data.steps.fol);
    renderPrologDetails(data.steps.prolog);

    // Render result
    renderResult(data);

    // Update status
    statusEl.textContent = 'Completed';
    statusEl.className = 'status success';

  } catch (error) {
    console.error('Query failed:', error);
    statusEl.textContent = 'Error: ' + error.message;
    statusEl.className = 'status error';
  }
}

// Reset pipeline
function resetPipeline() {
  ['stepWSD', 'stepFOL', 'stepProlog', 'stepResult'].forEach(id => {
    const step = document.getElementById(id);
    step.classList.remove('active', 'completed');
    step.querySelector('.step-status').textContent = '⏳';
  });
}

// Animate pipeline
async function animatePipeline(data) {
  // Step 1: WSD
  const stepWSD = document.getElementById('stepWSD');
  stepWSD.classList.add('active');
  stepWSD.querySelector('.step-status').textContent = '🔄';
  await sleep(1000);
  stepWSD.classList.remove('active');
  stepWSD.classList.add('completed');
  stepWSD.querySelector('.step-status').textContent = '✅';

  // Step 2: FOL
  const stepFOL = document.getElementById('stepFOL');
  stepFOL.classList.add('active');
  stepFOL.querySelector('.step-status').textContent = '🔄';
  await sleep(800);
  stepFOL.classList.remove('active');
  stepFOL.classList.add('completed');
  stepFOL.querySelector('.step-status').textContent = '✅';

  // Step 3: Prolog
  const stepProlog = document.getElementById('stepProlog');
  stepProlog.classList.add('active');
  stepProlog.querySelector('.step-status').textContent = '🔄';
  await sleep(1200);
  stepProlog.classList.remove('active');
  stepProlog.classList.add('completed');
  stepProlog.querySelector('.step-status').textContent = '✅';

  // Step 4: Result
  const stepResult = document.getElementById('stepResult');
  stepResult.classList.add('active');
  stepResult.querySelector('.step-status').textContent = '🔄';
  await sleep(500);
  stepResult.classList.remove('active');
  stepResult.classList.add('completed');
  stepResult.querySelector('.step-status').textContent = '✅';
}

// Render WSD details
function renderWSDDetails(wsdData) {
  const container = document.getElementById('wsdContent');

  // Get sentence count from first step (Sentence Tokenization)
  const sentenceStep = wsdData.steps.find(s => s.name === 'Sentence Tokenization');
  const sentenceCount = sentenceStep ? sentenceStep.output.length : 0;

  let html = `
    <div class="wsd-summary">
      <div class="summary-item"><strong>Method:</strong> ${wsdData.method.toUpperCase()}</div>
      <div class="summary-item"><strong>Sentences:</strong> ${sentenceCount}</div>
      <div class="summary-item"><strong>Content Words Disambiguated:</strong> ${wsdData.wsd_results ? wsdData.wsd_results.length : 0}</div>
    </div>`;

  // WSD Results Table (main output)
  if (wsdData.wsd_results && wsdData.wsd_results.length > 0) {
    html += `
    <div class="wsd-results-section">
      <h4 style="margin: 16px 0 8px; color: var(--accent);">WSD Results</h4>
      <table class="wsd-table">
        <thead>
          <tr>
            <th>Token</th>
            <th>POS</th>
            <th>Synset</th>
            <th>Definition</th>
            <th>Conf.</th>
          </tr>
        </thead>
        <tbody>
          ${wsdData.wsd_results.map(r => `
            <tr>
              <td><code>${r.token}</code></td>
              <td>${r.pos}</td>
              <td><code>${r.synset || 'N/A'}</code></td>
              <td class="definition-cell">${r.definition || '-'}</td>
              <td>${(r.confidence * 100).toFixed(0)}%</td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    </div>`;
  }

  // Collapsible Raw JSON
  html += `
    <div class="raw-json-section">
      <button class="toggle-json-btn" onclick="toggleRawJson('wsdRawJson')">
        📄 Show Raw JSON
      </button>
      <pre id="wsdRawJson" class="raw-json" style="display: none;">${JSON.stringify({
    method: wsdData.method,
    sentences: sentenceStep ? sentenceStep.output : [],
    wsd_results: wsdData.wsd_results
  }, null, 2)}</pre>
    </div>`;

  container.innerHTML = html;
}

// Toggle raw JSON visibility
function toggleRawJson(elementId) {
  const el = document.getElementById(elementId);
  const btn = el.previousElementSibling;
  if (el.style.display === 'none') {
    el.style.display = 'block';
    btn.textContent = '📄 Hide Raw JSON';
  } else {
    el.style.display = 'none';
    btn.textContent = '📄 Show Raw JSON';
  }
}

// Render FOL details
function renderFOLDetails(folData) {
  const container = document.getElementById('folContent');

  let html = `
    <div class="fol-summary">
      <div class="summary-item"><strong>Predicates:</strong> ${folData.predicates?.length || 0}</div>
      <div class="summary-item"><strong>Role Definitions:</strong> ${folData.role_definitions?.length || 0}</div>
      <div class="summary-item"><strong>Ground Formulas:</strong> ${folData.fol_formulas?.length || 0}</div>
    </div>`;

  // Display steps
  if (folData.steps && folData.steps.length > 0) {
    html += `<div class="fol-steps">`;
    folData.steps.forEach((step, idx) => {
      html += `
        <div class="fol-step">
          <div class="step-header">
            <span class="step-num">${step.step || idx + 1}</span>
            <span class="step-desc">${step.name}</span>
          </div>
          <div class="step-detail-text">${step.description}</div>`;

      if (step.output && step.output.length > 0) {
        if (typeof step.output[0] === 'string') {
          // String array - display as code tags
          html += `<div class="step-output"><code>${step.output.join('</code>, <code>')}</code></div>`;
        } else if (step.output[0].synset) {
          // Synset objects - display as compact table
          html += `
            <div class="step-output">
              <table class="synset-table">
                <thead><tr><th>Token</th><th>Synset</th><th>POS</th></tr></thead>
                <tbody>
                  ${step.output.map(s => `<tr><td><code>${s.token}</code></td><td><code>${s.synset}</code></td><td>${s.pos}</td></tr>`).join('')}
                </tbody>
              </table>
            </div>`;
        } else {
          // Other objects - display as JSON
          html += `<div class="step-output"><pre>${JSON.stringify(step.output, null, 2)}</pre></div>`;
        }
      }
      html += `</div>`;
    });
    html += `</div>`;
  }

  // Display role definitions with FOL formulas
  if (folData.role_definitions && folData.role_definitions.length > 0) {
    html += `
      <div class="fol-roles">
        <h4 style="margin: 16px 0 8px; color: var(--accent);">Role Definitions (Điều 3 Luật AI)</h4>
        <div class="role-list">
          ${folData.role_definitions.map(r => `
            <div class="role-item">
              <strong>${r.role}:</strong>
              <code class="fol-formula">${r.definition}</code>
            </div>
          `).join('')}
        </div>
      </div>`;
  }

  html += `
    <div class="raw-json-section">
      <button class="toggle-json-btn" onclick="toggleRawJson('folRawJson')">📄 Show Raw JSON</button>
      <pre id="folRawJson" class="raw-json" style="display: none;">${JSON.stringify(folData, null, 2)}</pre>
    </div>`;
  container.innerHTML = html;
}


// Render Prolog details
function renderPrologDetails(prologData) {
  const container = document.getElementById('prologContent');

  let html = `
    <div class="prolog-query">
      <strong>Query:</strong> <code>${prologData.query}</code>
    </div>`;

  if (prologData.steps && prologData.steps.length > 0) {
    html += `<div class="prolog-steps">`;
    prologData.steps.forEach((step, idx) => {
      html += `
        <div class="prolog-step">
          <div class="step-header">
            <span class="step-num">${step.step || idx + 1}</span>
            <span class="step-desc">${step.description}</span>
          </div>`;

      // Handle both 'rule' and 'rule_applied' property names
      const rule = step.rule_applied || step.rule;
      if (rule) {
        html += `<div class="step-rule"><strong>Rule:</strong> <code>${rule}</code></div>`;
      }

      if (step.facts_checked && step.facts_checked.length > 0) {
        html += `<div class="step-facts"><strong>Facts:</strong> ${step.facts_checked.map(f => `<code>${f}</code>`).join(', ')}</div>`;
      }

      if (step.candidates && step.candidates.length > 0) {
        html += `<div class="step-candidates"><strong>Candidates:</strong> ${step.candidates.map(c => `<code>${c}</code>`).join(', ')}</div>`;
      }

      if (step.result) {
        html += `<div class="step-result"><strong>Result:</strong> <span style="color: var(--success);">${step.result}</span></div>`;
      }

      html += `</div>`;
    });
    html += `</div>`;
  }

  html += `
    <div class="raw-json-section">
      <button class="toggle-json-btn" onclick="toggleRawJson('prologRawJson')">📄 Show Raw JSON</button>
      <pre id="prologRawJson" class="raw-json" style="display: none;">${JSON.stringify(prologData, null, 2)}</pre>
    </div>`;

  container.innerHTML = html;
}

// Render result
function renderResult(data) {
  const resultEl = document.getElementById('finalResult');

  const matchIcon = data.match ? '✓' : '✗';
  const matchClass = data.match ? 'match' : 'mismatch';

  resultEl.innerHTML = `
    <div class="result-comparison">
      <div class="result-item">
        <div class="result-label">Predicted Result</div>
        <div class="result-value predicted" style="color: ${data.success ? 'var(--success)' : 'var(--error)'};">${data.result}</div>
      </div>
      <div class="result-item">
        <div class="result-label">Ground Truth</div>
        <div class="result-value expected">${data.expected_result}</div>
      </div>
      <div class="result-item">
        <div class="result-label">Match</div>
        <div class="result-value ${matchClass}">${matchIcon} ${data.match ? 'Correct' : 'Incorrect'}</div>
      </div>
    </div>
  `;
}

// Toggle accordion
function toggleAccordion(id) {
  const item = document.getElementById(id).parentElement;
  item.classList.toggle('open');
}

// Utility
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}
