/**
 * Phases 15–22: Main application controller.
 * Wires together: WebSocket, audio engine, camera engine, UI panels.
 */

import { AudioRecorder, AudioPlayer } from './audio-engine.js';
import { CameraEngine } from './camera-engine.js';

// ============================================================
// State
// ============================================================
const API = window.location.origin;
const WS_URL = `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws/live`;

let ws = null;
let recorder = null;
let player = new AudioPlayer();
let camera = null;
let currentAgentMsg = null;
let reconnectTimer = null;
let isPaused = false;

// ============================================================
// DOM References
// ============================================================
const $ = (sel) => document.querySelector(sel);
const dom = {
    statusDot:   $('.status__dot'),
    statusText:  $('.status__text'),
    docCount:    $('#doc-count'),
    chat:        $('#chat'),
    textInput:   $('#text-input'),
    btnSend:     $('#btn-send'),
    btnMic:      $('#btn-mic'),
    btnPause:    $('#btn-pause'),
    btnCam:      $('#btn-cam'),
    uploadZone:  $('#upload-zone'),
    fileInput:   $('#file-input'),
    docList:     $('#doc-list'),
    emptyDocs:   $('#empty-docs'),
    citeList:    $('#cite-list'),
    emptyCites:  $('#empty-cites'),
    citeCount:   $('#cite-count'),
    cameraVideo: $('#camera-preview'),
    cameraOff:   $('#camera-off'),
    audit:       $('#audit'),
    waveform:    $('#waveform'),
};

// ============================================================
// WebSocket Connection (Phase 11)
// ============================================================
function connect() {
    if (ws && ws.readyState <= 1) return;

    ws = new WebSocket(WS_URL);
    ws.onopen = () => {
        setStatus('connected', 'Connected');
        addAudit('Session connected');
    };
    ws.onclose = () => {
        setStatus('off', 'Disconnected');
        addAudit('Disconnected');
        reconnectTimer = setTimeout(connect, 3000);
    };
    ws.onerror = () => setStatus('error', 'Connection error');
    ws.onmessage = (e) => handleMessage(JSON.parse(e.data));
}

function wsSend(data) {
    if (ws && ws.readyState === 1) ws.send(JSON.stringify(data));
}

// ============================================================
// Server Message Handler
// ============================================================
function handleMessage(msg) {
    switch (msg.type) {
        case 'audio':
            player.enqueue(b64toAB(msg.data));
            break;

        case 'transcript_output':
            updateAgentMsg(msg.text, msg.full_text);
            break;

        case 'transcript_input':
            addMsg('user', msg.text);
            addAudit(`User: "${trunc(msg.text, 60)}"`);
            break;

        case 'citations':
            showCitations(msg.citations);
            break;

        case 'turn_complete':
            if (msg.validation) addGroundingBadge(msg.validation);
            if (msg.follow_ups) renderFollowUps(msg.follow_ups);
            setStatus('connected', 'Connected');
            break;

        case 'interrupted':
            addAudit('Interrupted');
            break;

        case 'status':
            if (msg.message === 'connected') {
                addAudit('Gemini Live session active');
                setStatus('connected', 'Connected');
            } else if (msg.message === 'reconnecting') {
                addAudit('Session expired, auto-reconnecting...');
                setStatus('processing', 'Reconnecting...');
            }
            break;

        case 'error':
            addMsg('agent', `Error: ${msg.message}`);
            addAudit(`Error: ${msg.message}`);
            setStatus('error', 'Error');
            break;

        case 'pong':
            break;
    }
}

// ============================================================
// Chat (Phase 17)
// ============================================================
function addMsg(role, text) {
    const div = document.createElement('div');
    div.className = `msg msg--${role}`;
    div.innerHTML = `
        <div class="msg__label">${role === 'user' ? 'You' : 'GroundTruth'}</div>
        <div class="msg__bubble">${fmtCitations(esc(text))}</div>
    `;
    dom.chat.appendChild(div);
    dom.chat.scrollTop = dom.chat.scrollHeight;
    if (role === 'agent') currentAgentMsg = div;
}

function updateAgentMsg(newText, fullText) {
    if (!currentAgentMsg || currentAgentMsg.querySelector('.msg__label').textContent !== 'GroundTruth') {
        addMsg('agent', '');
    }
    currentAgentMsg.querySelector('.msg__bubble').innerHTML = fmtCitations(esc(fullText || newText));
    dom.chat.scrollTop = dom.chat.scrollHeight;
}

function addGroundingBadge(v) {
    if (!currentAgentMsg) return;
    const bubble = currentAgentMsg.querySelector('.msg__bubble');
    const statusMap = {
        grounded: ['grounded', '\u2713 Grounded'],
        partially_grounded: ['partial', '\u26A0 Partially grounded'],
        ungrounded: ['ungrounded', '\u2717 Ungrounded'],
        no_match: ['no-context', '\u2139 No matching content'],
        no_context: ['no-context', '\u2139 No documents'],
    };
    const [cls, label] = statusMap[v.status] || statusMap.no_context;

    // Footer container: badge + confidence meter side by side
    const footer = document.createElement('div');
    footer.className = 'msg__footer';

    // Grounding badge
    const badge = document.createElement('div');
    badge.className = `grounding-badge grounding-badge--${cls}`;
    badge.textContent = `${label}${v.cited_sources?.length ? ` (${v.cited_sources.length} sources)` : ''}`;
    footer.appendChild(badge);

    // Confidence Score Meter (radial gauge)
    if (v.confidence && v.confidence.score > 0) {
        footer.appendChild(createConfidenceMeter(v.confidence));
    }

    bubble.appendChild(footer);
    currentAgentMsg = null;
}

function createConfidenceMeter(conf) {
    const score = conf.score;
    const r = 18;
    const circ = 2 * Math.PI * r;  // ~113.1
    const offset = circ * (1 - score / 100);
    const color = score >= 75 ? 'var(--accent)' : score >= 45 ? 'var(--warning)' : 'var(--danger)';
    const level = score >= 75 ? 'High' : score >= 45 ? 'Medium' : 'Low';

    const cov = conf.breakdown?.citation_coverage ?? 0;
    const src = conf.breakdown?.source_quality ?? 0;
    const gnd = conf.breakdown?.grounding_status ?? 0;

    const wrapper = document.createElement('div');
    wrapper.className = 'confidence-wrapper';

    const meter = document.createElement('div');
    meter.className = 'confidence-meter';
    meter.innerHTML = `
        <svg viewBox="0 0 44 44" class="confidence-svg">
            <circle cx="22" cy="22" r="${r}" fill="none" stroke="var(--border)" stroke-width="3"/>
            <circle cx="22" cy="22" r="${r}" fill="none" stroke="${color}" stroke-width="3"
                    stroke-linecap="round" stroke-dasharray="${circ.toFixed(1)}"
                    stroke-dashoffset="${offset.toFixed(1)}" transform="rotate(-90 22 22)"
                    class="confidence-arc"/>
        </svg>
        <div class="confidence-text">
            <span class="confidence-value" style="color:${color}">${Math.round(score)}%</span>
            <span class="confidence-label">confidence</span>
        </div>
    `;
    wrapper.appendChild(meter);

    // Info button with tooltip breakdown
    const infoBtn = document.createElement('button');
    infoBtn.className = 'confidence-info-btn';
    infoBtn.textContent = 'i';
    infoBtn.setAttribute('aria-label', 'Confidence score breakdown');

    const tooltip = document.createElement('div');
    tooltip.className = 'confidence-tooltip';
    tooltip.innerHTML = `
        <div class="confidence-tooltip__title">${level} Confidence (${Math.round(score)}%)</div>
        <div class="confidence-tooltip__row">
            <span>Citation Coverage</span>
            <div class="confidence-tooltip__bar"><div class="confidence-tooltip__fill" style="width:${cov}%;background:${cov >= 60 ? 'var(--accent)' : cov >= 30 ? 'var(--warning)' : 'var(--danger)'}"></div></div>
            <span class="confidence-tooltip__val">${Math.round(cov)}%</span>
        </div>
        <div class="confidence-tooltip__row">
            <span>Source Quality</span>
            <div class="confidence-tooltip__bar"><div class="confidence-tooltip__fill" style="width:${src}%;background:${src >= 60 ? 'var(--accent)' : src >= 30 ? 'var(--warning)' : 'var(--danger)'}"></div></div>
            <span class="confidence-tooltip__val">${Math.round(src)}%</span>
        </div>
        <div class="confidence-tooltip__row">
            <span>Grounding Status</span>
            <div class="confidence-tooltip__bar"><div class="confidence-tooltip__fill" style="width:${gnd}%;background:${gnd >= 60 ? 'var(--accent)' : gnd >= 30 ? 'var(--warning)' : 'var(--danger)'}"></div></div>
            <span class="confidence-tooltip__val">${Math.round(gnd)}%</span>
        </div>
        <div class="confidence-tooltip__formula">C = 0.4\u00D7Coverage + 0.3\u00D7Sources + 0.3\u00D7Grounding</div>
    `;

    infoBtn.onclick = (e) => {
        e.stopPropagation();
        // Close any other open tooltips
        document.querySelectorAll('.confidence-tooltip.show').forEach(t => {
            if (t !== tooltip) t.classList.remove('show');
        });
        tooltip.classList.toggle('show');
    };

    wrapper.appendChild(infoBtn);
    wrapper.appendChild(tooltip);

    // Close tooltip when clicking elsewhere
    document.addEventListener('click', () => tooltip.classList.remove('show'), { once: true });

    return wrapper;
}

// ============================================================
// Smart Follow-up Suggestions
// ============================================================
function renderFollowUps(suggestions) {
    if (!suggestions || !suggestions.length) return;
    // Remove previous follow-up chips
    document.querySelectorAll('.follow-ups').forEach(el => el.remove());

    const container = document.createElement('div');
    container.className = 'follow-ups';

    const label = document.createElement('span');
    label.className = 'follow-ups__label';
    label.textContent = 'Follow up:';
    container.appendChild(label);

    suggestions.forEach(q => {
        const chip = document.createElement('button');
        chip.className = 'follow-up-chip';
        chip.textContent = q;
        chip.onclick = () => {
            container.remove();
            sendTextQuery(q);
        };
        container.appendChild(chip);
    });

    dom.chat.appendChild(container);
    dom.chat.scrollTop = dom.chat.scrollHeight;
}

async function sendTextQuery(text) {
    if (!text.trim()) return;
    addMsg('user', text);
    dom.textInput.value = '';
    setStatus('processing', 'Processing...');

    if (ws && ws.readyState === 1) {
        wsSend({ type: 'text', text });
        return;
    }

    // REST fallback
    try {
        const res = await fetch(`${API}/api/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: text }),
        });
        const data = await res.json();
        if (res.ok) {
            addMsg('agent', data.response);
            if (data.citations) showCitations(data.citations);
            if (data.validation) addGroundingBadge(data.validation);
        } else {
            addMsg('agent', `Error: ${data.detail || 'Unknown'}`);
        }
    } catch (err) {
        addMsg('agent', `Connection error: ${err.message}`);
    }
    setStatus('connected', 'Connected');
}

// ============================================================
// Audio (Phase 18-19) — Click-to-toggle mic + Pause
// ============================================================
async function toggleMic() {
    if (recorder && recorder.isRecording) {
        // Stop recording
        recorder.stop();
        recorder = null;
        isPaused = false;
        dom.btnMic.classList.remove('btn-circle--recording');
        dom.btnPause.style.display = 'none';
        dom.btnPause.classList.remove('btn-circle--active');
        clearWaveform();
        setStatus('connected', 'Connected');
        addAudit('Microphone off');
    } else {
        // Start recording
        recorder = new AudioRecorder(
            (pcm16) => {
                if (!isPaused) wsSend({ type: 'audio', data: abTo64(pcm16.buffer) });
            },
            (bars) => { if (!isPaused) renderWaveform(bars); },
        );
        const ok = await recorder.start();
        if (ok) {
            isPaused = false;
            dom.btnMic.classList.add('btn-circle--recording');
            dom.btnPause.style.display = '';
            setStatus('processing', 'Listening...');
            addAudit('Microphone on');
        }
    }
}

function togglePause() {
    if (!recorder || !recorder.isRecording) return;
    isPaused = !isPaused;
    if (isPaused) {
        dom.btnPause.classList.add('btn-circle--active');
        dom.btnPause.innerHTML = '&#x25b6;&#xfe0f;';
        dom.btnPause.title = 'Resume';
        clearWaveform();
        setStatus('connected', 'Paused');
        addAudit('Microphone paused');
    } else {
        dom.btnPause.classList.remove('btn-circle--active');
        dom.btnPause.innerHTML = '&#x23f8;&#xfe0f;';
        dom.btnPause.title = 'Pause';
        setStatus('processing', 'Listening...');
        addAudit('Microphone resumed');
    }
}

// ============================================================
// Camera (Phase 20)
// ============================================================
async function toggleCamera() {
    if (camera && camera.active) {
        camera.stop();
        camera = null;
        dom.cameraVideo.classList.remove('camera-preview--active');
        dom.cameraOff.style.display = '';
        dom.btnCam.classList.remove('btn-circle--active');
        addAudit('Camera off');
    } else {
        camera = new CameraEngine(
            dom.cameraVideo,
            (buf) => wsSend({ type: 'video', data: abTo64(buf) }),
        );
        const ok = await camera.start();
        if (ok) {
            dom.cameraVideo.classList.add('camera-preview--active');
            dom.cameraOff.style.display = 'none';
            dom.btnCam.classList.add('btn-circle--active');
            addAudit('Camera on (1 FPS)');
        }
    }
}

// ============================================================
// Citations (Phase 21)
// ============================================================
function showCitations(cites) {
    if (!cites || !cites.length) return;
    dom.emptyCites.style.display = 'none';
    dom.citeList.querySelectorAll('.cite-card').forEach(el => el.remove());

    cites.forEach(c => {
        const card = document.createElement('div');
        card.className = 'cite-card';
        card.innerHTML = `
            <div class="cite-card__head">
                <div class="cite-card__num">${c.index}</div>
                <div class="cite-card__source">${esc(c.doc_name)}</div>
                ${c.relevance_score ? `<div class="cite-card__score">${c.relevance_score.toFixed(2)}</div>` : ''}
            </div>
            <div class="cite-card__excerpt">${esc(c.excerpt)}</div>
        `;
        dom.citeList.appendChild(card);
    });
    dom.citeCount.textContent = `${cites.length} source${cites.length !== 1 ? 's' : ''}`;
}

// ============================================================
// Waveform Visualizer (Phase 22)
// ============================================================
function renderWaveform(bars) {
    const children = dom.waveform.children;
    for (let i = 0; i < Math.min(bars.length, children.length); i++) {
        const h = Math.max(4, bars[i] * 36);
        children[i].style.height = `${h}px`;
        children[i].classList.toggle('waveform__bar--active', bars[i] > 0.1);
    }
}

function clearWaveform() {
    for (const bar of dom.waveform.children) {
        bar.style.height = '4px';
        bar.classList.remove('waveform__bar--active');
    }
}

// ============================================================
// Document Upload (Phase 16)
// ============================================================
async function uploadFile(file) {
    const form = new FormData();
    form.append('file', file);
    try {
        const res = await fetch(`${API}/api/documents/upload`, { method: 'POST', body: form });
        const data = await res.json();
        if (res.ok) {
            addDocItem(data);
            refreshDocCount();
            addAudit(`Uploaded: ${data.name} (${data.chunks} chunks)`);
            // Push document content into the active Gemini session
            wsSend({ type: 'doc_update' });
            // Fetch document health score (async, non-blocking)
            fetchDocHealth(data.doc_id);
        } else {
            alert(`Upload failed: ${data.detail}`);
        }
    } catch (err) {
        alert(`Upload error: ${err.message}`);
    }
}

async function loadDocuments() {
    try {
        const res = await fetch(`${API}/api/documents`);
        const data = await res.json();
        data.documents.forEach(d => addDocItem(d));
        refreshDocCount();
        // Fetch health scores for all docs
        if (data.documents.length > 0) fetchAllDocHealth();
    } catch (e) { /* ignore on first load */ }
}

function addDocItem(doc) {
    dom.emptyDocs.style.display = 'none';
    const icons = { pdf: '\u{1F4D5}', docx: '\u{1F4D8}', txt: '\u{1F4C4}', md: '\u{1F4DD}' };
    const ext = doc.name.split('.').pop().toLowerCase();
    const item = document.createElement('div');
    item.className = 'doc-item';
    item.dataset.docId = doc.doc_id;
    item.innerHTML = `
        <div class="doc-item__icon">${icons[ext] || '\u{1F4C4}'}</div>
        <div class="doc-item__info">
            <div class="doc-item__name">${esc(doc.name)}</div>
            <div class="doc-item__meta">${doc.chunks} chunks \u00B7 ${(doc.content_length/1024).toFixed(1)}KB</div>
            <div class="doc-health-bar"><div class="doc-health-bar__fill"></div></div>
        </div>
        <button class="doc-item__remove" title="Remove">\u00D7</button>
    `;
    item.querySelector('.doc-item__remove').onclick = () => removeDoc(doc.doc_id, item);
    dom.docList.appendChild(item);
}

async function removeDoc(id, el) {
    await fetch(`${API}/api/documents/${id}`, { method: 'DELETE' });
    el.remove();
    refreshDocCount();
    if (!dom.docList.querySelector('.doc-item')) dom.emptyDocs.style.display = '';
    addAudit(`Removed document ${id}`);
}

function refreshDocCount() {
    const n = dom.docList.querySelectorAll('.doc-item').length;
    dom.docCount.textContent = `${n} doc${n !== 1 ? 's' : ''}`;
}

// ============================================================
// Document Health Score
// ============================================================
async function fetchDocHealth(docId) {
    try {
        const res = await fetch(`${API}/api/documents/${docId}/health`);
        if (res.ok) {
            const health = await res.json();
            updateDocHealth(docId, health);
        }
    } catch (e) { /* non-critical */ }
}

async function fetchAllDocHealth() {
    try {
        const res = await fetch(`${API}/api/documents/health`);
        if (res.ok) {
            const data = await res.json();
            for (const [docId, health] of Object.entries(data.scores)) {
                updateDocHealth(docId, health);
            }
        }
    } catch (e) { /* non-critical */ }
}

function updateDocHealth(docId, health) {
    const item = dom.docList.querySelector(`[data-doc-id="${docId}"]`);
    if (!item) return;

    const fill = item.querySelector('.doc-health-bar__fill');
    if (!fill) return;

    const score = health.overall_score || 0;
    const level = health.level || (score >= 80 ? 'excellent' : score >= 60 ? 'good' : score >= 40 ? 'fair' : 'poor');
    const color = score >= 80 ? 'var(--accent)' : score >= 60 ? '#4ade80' : score >= 40 ? 'var(--warning)' : 'var(--danger)';
    fill.style.width = `${score}%`;
    fill.style.background = color;
    fill.title = `Health: ${Math.round(score)}% (${level})\nRichness: ${health.breakdown?.content_richness ?? '—'}%\nDiversity: ${health.breakdown?.keyword_diversity ?? '—'}%\nEmbeddings: ${health.breakdown?.embedding_coverage ?? '—'}%\nStructure: ${health.breakdown?.structure_quality ?? '—'}%`;

    // Update meta text to include health
    const meta = item.querySelector('.doc-item__meta');
    if (meta && !meta.dataset.healthSet) {
        const label = score >= 80 ? 'Excellent' : score >= 60 ? 'Good' : score >= 40 ? 'Fair' : 'Low';
        meta.textContent += ` \u00B7 ${Math.round(score)}% ${label}`;
        meta.dataset.healthSet = '1';
    }
}

// ============================================================
// Audit Trail (Phase 22)
// ============================================================
function addAudit(text) {
    const entry = document.createElement('div');
    entry.className = 'audit__item';
    entry.innerHTML = `<span class="audit__time">${new Date().toLocaleTimeString()}</span> ${esc(text)}`;
    dom.audit.prepend(entry);
    // Keep bounded
    while (dom.audit.children.length > 100) dom.audit.lastChild.remove();
}

// ============================================================
// Status
// ============================================================
function setStatus(state, text) {
    dom.statusDot.className = `status__dot${state === 'connected' ? ' status__dot--connected' : state === 'error' ? ' status__dot--error' : state === 'processing' ? ' status__dot--processing' : ''}`;
    dom.statusText.textContent = text;
}

// ============================================================
// Utilities
// ============================================================
function esc(t) {
    const d = document.createElement('div');
    d.textContent = t;
    return d.innerHTML;
}

function fmtCitations(t) {
    // Highlight all "Source N" references regardless of bracket/comma format
    return t.replace(/Source\s+(\d+)/g, '<span class="citation-ref">Source $1</span>');
}

function trunc(s, n) { return s.length > n ? s.slice(0, n) + '...' : s; }

function abTo64(buf) {
    const bytes = new Uint8Array(buf);
    let bin = '';
    for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
    return btoa(bin);
}

function b64toAB(b64) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return bytes.buffer;
}

// ============================================================
// Event Listeners
// ============================================================
dom.btnSend.addEventListener('click', () => sendTextQuery(dom.textInput.value));
dom.textInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') sendTextQuery(dom.textInput.value); });

// Mic: click to toggle
dom.btnMic.addEventListener('click', toggleMic);

// Pause: click to pause/resume
dom.btnPause.addEventListener('click', togglePause);

dom.btnCam.addEventListener('click', toggleCamera);

// File upload
dom.uploadZone.addEventListener('click', () => dom.fileInput.click());
dom.uploadZone.addEventListener('dragover', (e) => { e.preventDefault(); dom.uploadZone.classList.add('upload--dragover'); });
dom.uploadZone.addEventListener('dragleave', () => dom.uploadZone.classList.remove('upload--dragover'));
dom.uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dom.uploadZone.classList.remove('upload--dragover');
    [...e.dataTransfer.files].forEach(uploadFile);
});
dom.fileInput.addEventListener('change', () => {
    [...dom.fileInput.files].forEach(uploadFile);
    dom.fileInput.value = '';
});

// ============================================================
// Initialize
// ============================================================
connect();
loadDocuments();
addAudit('GroundTruth initialized');
