// NeuralDBG Aquarium Preview Logic
// Author: Antigravity

document.addEventListener('DOMContentLoaded', () => {
    updateTimestamp();
    loadCausalPackage();
    
    // Auto-refresh every 30s to simulate live monitoring
    setInterval(loadCausalPackage, 30000);
});

async function loadCausalPackage() {
    const telemetryLog = document.getElementById('telemetry-log');
    const insightsContent = document.getElementById('insights-content');
    
    addLogEntry('INGESTING_CAUSAL_PACKAGE...', 'system');
    
    try {
        // Attempt to load the package from the standard output directory
        // Note: In a real preview, this would be served via a local server
        const response = await fetch('../outputs/aquarium/events.json');
        if (!response.ok) throw new Error('PACKAGE_NOT_FOUND');
        
        const data = await response.json();
        renderDashboard(data);
        addLogEntry('SYNC_COMPLETE: ' + data.events.length + ' EVENTS INGESTED', 'system');
        
    } catch (err) {
        addLogEntry('ERROR: ' + err.message, 'error');
        console.error('Failed to load causal package:', err);
    }
}

function renderDashboard(data) {
    // 1. Update Metadata
    document.getElementById('current-step').textContent = data.step;
    document.getElementById('model-type').textContent = data.metadata ? data.metadata.model_type : 'N/A';
    
    // 2. Render Graph (Mermaid)
    renderMermaidGraph(data);
    
    // 3. Render Insights
    renderInsights(data.hypotheses);
}

function renderMermaidGraph(data) {
    const container = document.getElementById('graph-viz');
    
    // Construct a simple Mermaid flow from events
    let graphDef = 'graph LR\n';
    
    // Add unique layers as nodes
    const layers = [...new Set(data.events.map(e => e.layer_name))];
    layers.forEach(l => {
        graphDef += `  ${l.replace(/[^a-zA-Z]/g, '')}[${l}]\n`;
    });
    
    // Add causal connections from hypotheses if available
    data.hypotheses.forEach(h => {
        if (h.causal_chain.length > 1) {
            for (let i = 0; i < h.causal_chain.length - 1; i++) {
                const from = h.causal_chain[i].replace(/[^a-zA-Z]/g, '');
                const to = h.causal_chain[i+1].replace(/[^a-zA-Z]/g, '');
                graphDef += `  ${from} ==CAUSAL==> ${to}\n`;
            }
        }
    });

    container.innerHTML = `<pre class="mermaid">${graphDef}</pre>`;
    
    // Re-initialize mermaid
    if (window.mermaid) {
        window.mermaid.init(undefined, ".mermaid");
    }
}

function renderInsights(hypotheses) {
    const list = document.getElementById('insights-content');
    list.innerHTML = '';
    
    if (hypotheses.length === 0) {
        list.innerHTML = '<div class="insight-card empty"><p>NO_FAILURE_DETECTED</p></div>';
        return;
    }
    
    hypotheses.forEach(h => {
        const card = document.createElement('div');
        const level = h.confidence > 0.8 ? 'CRITICAL' : 'WARNING';
        card.className = `insight-card ${level}`;
        
        card.innerHTML = `
            <div class="level">${level} [CONFIDENCE: ${(h.confidence * 100).toFixed(1)}%]</div>
            <h3>${h.description}</h3>
            <p style="font-size: 0.7rem; margin-top: 10px; opacity: 0.7">CAUSAL_CHAIN: ${h.causal_chain.join(' -> ')}</p>
        `;
        list.appendChild(card);
    });
}

function addLogEntry(text, type = '') {
    const log = document.getElementById('telemetry-log');
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    const time = new Date().toLocaleTimeString('en-GB', { hour12: false });
    entry.textContent = `[${time}] ${text}`;
    log.prepend(entry);
}

function updateTimestamp() {
    document.getElementById('timestamp').textContent = 'TRACE_LOCAL_TIME: ' + new Date().toISOString();
}
