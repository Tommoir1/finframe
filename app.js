import { ZipStore, csvText, cocoBoxFromNormalized, yoloLineFromNormalized } from './dataset-utils.js';

(() => {
  'use strict';

  const STORAGE_KEY = 'finframe-project-v1';
  const colors = ['#ff8465', '#41bda8', '#efb24e', '#5d9bd3', '#c47ccc', '#77ad5c', '#e0648d'];
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const uid = (prefix = 'id') => `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 7)}`;

  const starterSpecies = [
    { id: 'sp_snapper', common: 'Australasian snapper', scientific: 'Chrysophrys auratus', code: 'CHRAUR', color: '#ff8465' },
    { id: 'sp_trevally', common: 'Silver trevally', scientific: 'Pseudocaranx georgianus', code: 'PSEGE', color: '#41bda8' },
    { id: 'sp_wrasse', common: 'Blue-throated wrasse', scientific: 'Notolabrus tetricus', code: 'NOTTET', color: '#efb24e' },
    { id: 'sp_leatherjacket', common: 'Horseshoe leatherjacket', scientific: 'Meuschenia hippocrepis', code: 'MEUHIPP', color: '#5d9bd3' }
  ];

  function emptyProject() {
    return {
      schemaVersion: '1.0',
      project: { name: 'Untitled BRUV survey', deploymentId: '', site: '', observer: '', date: '', depth: '', notes: '', createdAt: new Date().toISOString(), updatedAt: new Date().toISOString() },
      video: { name: '', duration: 0, width: 1920, height: 1080, fps: 25 },
      species: structuredClone(starterSpecies),
      frames: [],
      activeSpeciesId: starterSpecies[0].id
    };
  }

  function restoreProject() {
    try {
      const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
      if (saved?.schemaVersion && Array.isArray(saved.frames) && Array.isArray(saved.species)) return saved;
    } catch (_) { /* ignored */ }
    return emptyProject();
  }

  let state = restoreProject();
  let currentTime = 0;
  let currentFrameKey = null;
  let draft = [];
  let selectedId = null;
  let activeTool = 'draw';
  let pointerAction = null;
  let videoUrl = null;
  let demoMode = false;
  let demoPlaying = false;
  let demoTimer = null;
  let reviewFilter = 'all';
  let toastTimer = null;

  const els = {
    video: $('#video'), canvas: $('#annotationCanvas'), stage: $('#videoStage'), empty: $('#stageEmpty'), demoScene: $('#demoScene'),
    timeline: $('#timeline'), playBtn: $('#playBtn'), currentTime: $('#currentTime'), duration: $('#duration'), frameNumber: $('#frameNumber'),
    fpsInput: $('#fpsInput'), speedSelect: $('#speedSelect'), speciesList: $('#speciesList'), speciesSearch: $('#speciesSearch'),
    annotationList: $('#annotationList'), frameNote: $('#frameNote'), frameFishCount: $('#frameFishCount'), inspectorTime: $('#inspectorTime'),
    frameState: $('#frameState'), speciesHud: $('#speciesHud'), maxnList: $('#maxnList'), totalMaxN: $('#totalMaxN'), toast: $('#toast')
  };

  function saveProject(message = 'Saved locally') {
    state.project.updatedAt = new Date().toISOString();
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
      $('#saveStatus').textContent = message;
      $('.save-dot').style.background = '#56a981';
    } catch (_) {
      $('#saveStatus').textContent = 'Storage unavailable';
      $('.save-dot').style.background = '#f47453';
    }
  }

  function toast(message) {
    els.toast.textContent = message;
    els.toast.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => els.toast.classList.remove('show'), 2200);
  }

  function safeFileName(value) {
    return (value || 'finframe_project').trim().replace(/[^a-z0-9-_]+/gi, '_').replace(/^_+|_+$/g, '').toLowerCase();
  }

  function formatTime(seconds) {
    if (!Number.isFinite(seconds)) seconds = 0;
    const ms = Math.floor((seconds % 1) * 1000).toString().padStart(3, '0');
    const whole = Math.floor(seconds);
    const s = (whole % 60).toString().padStart(2, '0');
    const m = (Math.floor(whole / 60) % 60).toString().padStart(2, '0');
    const h = Math.floor(whole / 3600);
    return `${h ? `${h.toString().padStart(2, '0')}:` : ''}${m}:${s}.${ms}`;
  }

  function currentKey() { return Math.round(currentTime * state.video.fps); }
  function getFrame(key = currentKey()) { return state.frames.find(frame => frame.frameNumber === key); }
  function speciesById(id) { return state.species.find(item => item.id === id) || state.species[0]; }
  function hasMedia() { return demoMode || Boolean(videoUrl); }

  function selectView(name) {
    $$('.nav-item').forEach(btn => btn.classList.toggle('is-active', btn.dataset.view === name));
    $$('.view').forEach(view => view.classList.remove('is-visible'));
    $(`#${name}View`).classList.add('is-visible');
    if (name === 'review') renderReview();
    if (name === 'dataset') renderDataset();
    if (name === 'annotate') setTimeout(resizeCanvas, 30);
  }

  function renderSpecies() {
    const q = els.speciesSearch.value.trim().toLowerCase();
    const items = state.species.filter(sp => `${sp.common} ${sp.scientific} ${sp.code}`.toLowerCase().includes(q));
    els.speciesList.innerHTML = items.map(sp => `
      <button class="species-item ${state.activeSpeciesId === sp.id ? 'is-active' : ''}" data-species-id="${sp.id}">
        <span class="species-swatch" style="background:${sp.color}"></span>
        <span class="species-copy"><strong>${escapeHtml(sp.common)}</strong><small>${escapeHtml(sp.scientific || 'Unspecified taxon')}</small></span>
        <span class="species-code">${escapeHtml(sp.code)}</span>
      </button>`).join('') || '<div class="annotation-empty"><strong>No matches</strong><p>Add a new species or try another search.</p></div>';
    $$('.species-item', els.speciesList).forEach(btn => btn.addEventListener('click', () => {
      state.activeSpeciesId = btn.dataset.speciesId;
      saveProject(); renderSpecies(); drawCanvas(); renderHud();
    }));
  }

  function escapeHtml(value = '') {
    return String(value).replace(/[&<>'"]/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[char]));
  }

  function renderProjectInfo() {
    $('#projectName').textContent = state.project.name;
    $('#projectSubtitle').textContent = state.video.name || 'No video selected';
    $('#detailsSummary').textContent = [state.project.site, state.project.observer].filter(Boolean).join(' · ') || 'Add site and observer';
    els.fpsInput.value = state.video.fps || 25;
    $('#reviewCount').textContent = state.frames.length;
    $('#videoStatus').textContent = hasMedia() ? 'Ready' : 'Waiting';
    $('#videoStatus').classList.toggle('ready', hasMedia());
  }

  function loadFrameAtTime(force = false) {
    const key = currentKey();
    if (!force && key === currentFrameKey) return;
    currentFrameKey = key;
    const frame = getFrame(key);
    draft = frame ? structuredClone(frame.annotations) : [];
    selectedId = draft.some(item => item.id === selectedId) ? selectedId : null;
    els.frameNote.value = frame?.note || '';
    renderFrame();
  }

  function syncFrame({ createEmpty = false } = {}) {
    const key = currentKey();
    let frame = getFrame(key);
    if (!frame && (draft.length || createEmpty || els.frameNote.value.trim())) {
      frame = { id: uid('frame'), time: currentTime, frameNumber: key, annotations: [], note: '', reviewed: false, savedAt: new Date().toISOString() };
      state.frames.push(frame);
    }
    if (frame) {
      frame.time = currentTime;
      frame.annotations = structuredClone(draft);
      frame.note = els.frameNote.value.trim();
      frame.savedAt = new Date().toISOString();
      if (!draft.length && !frame.note && !createEmpty) state.frames = state.frames.filter(item => item.id !== frame.id);
    }
    state.frames.sort((a, b) => a.frameNumber - b.frameNumber);
    saveProject();
    renderFrame();
    renderMaxN();
    $('#reviewCount').textContent = state.frames.length;
  }

  function renderFrame() {
    els.currentTime.textContent = formatTime(currentTime);
    els.inspectorTime.textContent = formatTime(currentTime);
    els.duration.textContent = formatTime(state.video.duration || 0);
    els.frameNumber.textContent = `Frame ${currentKey().toLocaleString()}`;
    els.frameFishCount.textContent = draft.length;
    const frame = getFrame();
    els.frameState.textContent = draft.length ? `${draft.length} ${draft.length === 1 ? 'box' : 'boxes'} saved` : 'Frame not annotated';
    $('.live-dot').classList.toggle('active', draft.length > 0);
    $('#markReviewedBtn').textContent = frame?.reviewed ? '✓ Reviewed' : '✓ Mark reviewed';
    renderAnnotations();
    renderFrameCounts();
    renderHud();
    drawCanvas();
  }

  function renderAnnotations() {
    if (!draft.length) {
      els.annotationList.innerHTML = `<div class="annotation-empty"><span class="mini-box"></span><strong>No boxes on this frame</strong><p>Choose a species, then drag a box around each visible fish.</p></div>`;
      return;
    }
    els.annotationList.innerHTML = draft.map((ann, index) => {
      const sp = speciesById(ann.speciesId);
      return `<div class="annotation-card ${selectedId === ann.id ? 'is-selected' : ''}" data-ann-id="${ann.id}">
        <div class="annotation-card-head"><span class="annotation-dot" style="background:${sp.color}"></span><strong>${escapeHtml(sp.common)}</strong><span class="track-chip">${escapeHtml(ann.trackId)}</span><button class="delete-box" data-delete-id="${ann.id}" aria-label="Delete box">×</button></div>
        <div class="annotation-fields">
          <label>Species<select data-field="speciesId">${state.species.map(s => `<option value="${s.id}" ${s.id === ann.speciesId ? 'selected' : ''}>${escapeHtml(s.common)}</option>`).join('')}</select></label>
          <label>Track ID<input data-field="trackId" value="${escapeHtml(ann.trackId)}"></label>
          <label>Stage<select data-field="stage"><option ${ann.stage === 'Adult' ? 'selected' : ''}>Adult</option><option ${ann.stage === 'Juvenile' ? 'selected' : ''}>Juvenile</option><option ${ann.stage === 'Unknown' ? 'selected' : ''}>Unknown</option></select></label>
          <label>Activity<select data-field="activity"><option ${ann.activity === 'Passing' ? 'selected' : ''}>Passing</option><option ${ann.activity === 'Feeding' ? 'selected' : ''}>Feeding</option><option ${ann.activity === 'Schooling' ? 'selected' : ''}>Schooling</option><option ${ann.activity === 'Resting' ? 'selected' : ''}>Resting</option></select></label>
        </div>
        <div class="flag-row"><span>Box ${index + 1} · ${Math.round(ann.w * state.video.width)} × ${Math.round(ann.h * state.video.height)} px</span><label class="switch-label"><input type="checkbox" data-field="uncertain" ${ann.uncertain ? 'checked' : ''}> Uncertain</label></div>
      </div>`;
    }).join('');
    $$('.annotation-card', els.annotationList).forEach(card => {
      card.addEventListener('click', event => {
        if (event.target.closest('input,select,button')) return;
        selectedId = card.dataset.annId; renderFrame();
      });
      $$('[data-field]', card).forEach(control => control.addEventListener('change', () => {
        const ann = draft.find(item => item.id === card.dataset.annId);
        const field = control.dataset.field;
        ann[field] = control.type === 'checkbox' ? control.checked : control.value;
        if (field === 'speciesId') state.activeSpeciesId = control.value;
        syncFrame(); renderSpecies();
      }));
    });
    $$('[data-delete-id]', els.annotationList).forEach(btn => btn.addEventListener('click', () => deleteAnnotation(btn.dataset.deleteId)));
  }

  function renderHud() {
    const counts = new Map();
    draft.forEach(ann => counts.set(ann.speciesId, (counts.get(ann.speciesId) || 0) + 1));
    const maxBySpecies = new Map(summaryForSpecies().map(row => [row.species.id, row.maxN]));
    els.speciesHud.innerHTML = [...counts].map(([id, count]) => {
      const sp = speciesById(id); const isMax = count > 0 && count === maxBySpecies.get(id);
      return `<span class="hud-chip"><i style="background:${sp.color}"></i>${escapeHtml(sp.code)} · ${count}${isMax ? ' · MAXN' : ''}</span>`;
    }).join('');
  }

  function renderFrameCounts() {
    const counts = new Map();
    draft.forEach(ann => counts.set(ann.speciesId, (counts.get(ann.speciesId) || 0) + 1));
    const maxBySpecies = new Map(summaryForSpecies().map(row => [row.species.id, row.maxN]));
    $('#frameCountSummary').innerHTML = counts.size ? [...counts].map(([id, count]) => {
      const sp = speciesById(id); const isMax = count > 0 && count === maxBySpecies.get(id);
      return `<span class="frame-count-chip"><i style="background:${sp.color}"></i>${escapeHtml(sp.code)} <strong>${count}</strong>${isMax ? '<em>MAXN</em>' : ''}</span>`;
    }).join('') : '<span class="frame-count-empty">Per-species counts will appear here.</span>';
  }

  function summaryForSpecies() {
    return state.species.map(sp => {
      let maxN = 0, maxFrame = null, total = 0, nonzero = 0, first = null;
      const tracks = new Set();
      state.frames.forEach(frame => {
        const anns = frame.annotations.filter(ann => ann.speciesId === sp.id);
        const count = anns.length;
        if (count) {
          total += count; nonzero += 1;
          if (first === null || frame.time < first) first = frame.time;
          anns.forEach(ann => tracks.add(ann.trackId));
        }
        if (count > maxN) { maxN = count; maxFrame = frame; }
      });
      return { species: sp, maxN, maxFrame, mean: nonzero ? total / nonzero : 0, first, tracks: tracks.size, boxes: total };
    }).filter(row => row.boxes > 0);
  }

  function renderMaxN() {
    const rows = summaryForSpecies().sort((a, b) => b.maxN - a.maxN);
    const total = rows.reduce((sum, row) => sum + row.maxN, 0);
    els.totalMaxN.textContent = total;
    els.maxnList.innerHTML = rows.length ? rows.map(row => `<div class="maxn-item"><div class="maxn-species"><span class="species-swatch" style="background:${row.species.color}"></span><span class="maxn-copy"><strong>${escapeHtml(row.species.common)}</strong><small>Peak at ${formatTime(row.maxFrame.time)} · mean ${row.mean.toFixed(1)}</small></span></div><span class="maxn-value">${row.maxN}</span></div>`).join('') : '<div class="annotation-empty"><strong>No MaxN yet</strong><p>Draw boxes on at least one frame to start the live summary.</p></div>';
  }

  function getVideoBox() {
    const cw = els.canvas.clientWidth, ch = els.canvas.clientHeight;
    const vw = state.video.width || 1920, vh = state.video.height || 1080;
    const scale = Math.min(cw / vw, ch / vh);
    const width = vw * scale, height = vh * scale;
    return { x: (cw - width) / 2, y: (ch - height) / 2, width, height };
  }

  function resizeCanvas() {
    const rect = els.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    els.canvas.width = Math.max(1, Math.round(rect.width * dpr));
    els.canvas.height = Math.max(1, Math.round(rect.height * dpr));
    drawCanvas();
  }

  function drawCanvas() {
    const ctx = els.canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, els.canvas.clientWidth, els.canvas.clientHeight);
    if (!hasMedia()) return;
    const box = getVideoBox();
    draft.forEach(ann => {
      const sp = speciesById(ann.speciesId);
      const x = box.x + ann.x * box.width, y = box.y + ann.y * box.height, w = ann.w * box.width, h = ann.h * box.height;
      const selected = ann.id === selectedId;
      ctx.save();
      ctx.strokeStyle = sp.color; ctx.lineWidth = selected ? 3 : 2; ctx.setLineDash(ann.uncertain ? [7, 4] : []);
      ctx.fillStyle = `${sp.color}1f`; ctx.fillRect(x, y, w, h); ctx.strokeRect(x, y, w, h);
      ctx.setLineDash([]); ctx.font = '700 10px Inter, system-ui, sans-serif';
      const label = `${sp.code} · ${ann.trackId}`; const tw = ctx.measureText(label).width + 12;
      ctx.fillStyle = sp.color; ctx.fillRect(x, Math.max(box.y, y - 20), tw, 20);
      ctx.fillStyle = '#071713'; ctx.fillText(label, x + 6, Math.max(box.y + 13, y - 6));
      if (selected) {
        ctx.fillStyle = '#fff'; ctx.strokeStyle = sp.color; ctx.lineWidth = 2;
        [[x,y],[x+w,y],[x+w,y+h],[x,y+h]].forEach(([hx,hy]) => { ctx.beginPath(); ctx.rect(hx-4,hy-4,8,8); ctx.fill(); ctx.stroke(); });
      }
      ctx.restore();
    });
    if (pointerAction?.type === 'draw') {
      const sp = speciesById(state.activeSpeciesId); const rect = normaliseRect(pointerAction.start, pointerAction.current);
      ctx.save(); ctx.strokeStyle = sp.color; ctx.lineWidth = 2; ctx.setLineDash([5,4]);
      ctx.strokeRect(box.x + rect.x * box.width, box.y + rect.y * box.height, rect.w * box.width, rect.h * box.height); ctx.restore();
    }
  }

  function eventPoint(event) {
    const rect = els.canvas.getBoundingClientRect(); const box = getVideoBox();
    const px = event.clientX - rect.left, py = event.clientY - rect.top;
    return { x: Math.max(0, Math.min(1, (px - box.x) / box.width)), y: Math.max(0, Math.min(1, (py - box.y) / box.height)), px, py, inside: px >= box.x && px <= box.x + box.width && py >= box.y && py <= box.y + box.height };
  }
  function normaliseRect(a, b) { return { x: Math.min(a.x,b.x), y: Math.min(a.y,b.y), w: Math.abs(a.x-b.x), h: Math.abs(a.y-b.y) }; }
  function hitAnnotation(point) { return [...draft].reverse().find(ann => point.x >= ann.x && point.x <= ann.x+ann.w && point.y >= ann.y && point.y <= ann.y+ann.h); }
  function nextTrackId(speciesId) {
    const sp = speciesById(speciesId); let max = 0;
    state.frames.flatMap(f => f.annotations).forEach(ann => { if (ann.speciesId === speciesId) { const n = Number((ann.trackId.match(/(\d+)$/) || [])[1]); if (n > max) max = n; } });
    draft.forEach(ann => { if (ann.speciesId === speciesId) { const n = Number((ann.trackId.match(/(\d+)$/) || [])[1]); if (n > max) max = n; } });
    return `${sp.code}-${String(max + 1).padStart(3, '0')}`;
  }

  function onPointerDown(event) {
    if (!hasMedia()) return;
    const point = eventPoint(event); if (!point.inside) return;
    els.canvas.setPointerCapture(event.pointerId);
    if (activeTool === 'draw') {
      pointerAction = { type: 'draw', start: point, current: point };
    } else {
      const hit = hitAnnotation(point);
      if (!hit) { selectedId = null; renderFrame(); return; }
      selectedId = hit.id;
      const nearCorner = Math.abs(point.x - (hit.x + hit.w)) < .025 && Math.abs(point.y - (hit.y + hit.h)) < .025;
      pointerAction = { type: nearCorner ? 'resize' : 'move', start: point, origin: structuredClone(hit), annId: hit.id };
      renderFrame();
    }
  }
  function onPointerMove(event) {
    if (!pointerAction) return;
    const point = eventPoint(event);
    if (pointerAction.type === 'draw') pointerAction.current = point;
    else {
      const ann = draft.find(item => item.id === pointerAction.annId); if (!ann) return;
      const dx = point.x - pointerAction.start.x, dy = point.y - pointerAction.start.y;
      if (pointerAction.type === 'move') { ann.x = Math.max(0, Math.min(1-ann.w, pointerAction.origin.x+dx)); ann.y = Math.max(0, Math.min(1-ann.h, pointerAction.origin.y+dy)); }
      else { ann.w = Math.max(.01, Math.min(1-ann.x, pointerAction.origin.w+dx)); ann.h = Math.max(.01, Math.min(1-ann.y, pointerAction.origin.h+dy)); }
    }
    drawCanvas();
  }
  function onPointerUp(event) {
    if (!pointerAction) return;
    if (pointerAction.type === 'draw') {
      const rect = normaliseRect(pointerAction.start, eventPoint(event));
      if (rect.w > .008 && rect.h > .008) {
        const ann = { id: uid('box'), speciesId: state.activeSpeciesId, trackId: nextTrackId(state.activeSpeciesId), x: rect.x, y: rect.y, w: rect.w, h: rect.h, stage: 'Adult', activity: 'Passing', uncertain: false };
        draft.push(ann); selectedId = ann.id;
      }
    }
    pointerAction = null; syncFrame();
  }

  function deleteAnnotation(id) { draft = draft.filter(item => item.id !== id); if (selectedId === id) selectedId = null; syncFrame(); }

  function setTool(tool) {
    activeTool = tool;
    $$('[data-tool]').forEach(btn => btn.classList.toggle('is-active', btn.dataset.tool === tool));
    els.stage.classList.toggle('selecting', tool === 'select');
  }

  function setTime(value) {
    const duration = state.video.duration || 0;
    currentTime = Math.max(0, Math.min(duration, value));
    if (!demoMode && videoUrl) els.video.currentTime = currentTime;
    els.timeline.value = currentTime;
    loadFrameAtTime(true);
  }

  function stepFrame(direction) { pauseMedia(); setTime(currentTime + direction / state.video.fps); }
  function pauseMedia() { if (demoMode) { demoPlaying = false; clearInterval(demoTimer); } else els.video.pause(); els.playBtn.textContent = '▶'; els.playBtn.setAttribute('aria-label','Play'); }
  function togglePlayback() {
    if (!hasMedia()) { toast('Open a video or sample survey first'); return; }
    if (demoMode) {
      if (demoPlaying) return pauseMedia();
      demoPlaying = true; els.playBtn.textContent = 'Ⅱ'; els.playBtn.setAttribute('aria-label','Pause');
      const started = performance.now(), from = currentTime;
      demoTimer = setInterval(() => { currentTime = from + (performance.now()-started)/1000 * Number(els.speedSelect.value); if (currentTime >= state.video.duration) { setTime(0); pauseMedia(); } else { els.timeline.value = currentTime; loadFrameAtTime(); updateTimeLabels(); } }, 40);
    } else if (els.video.paused) els.video.play(); else els.video.pause();
  }
  function updateTimeLabels() { els.currentTime.textContent = formatTime(currentTime); els.inspectorTime.textContent = formatTime(currentTime); els.frameNumber.textContent = `Frame ${currentKey().toLocaleString()}`; }

  function loadVideo(file) {
    if (!file) return;
    const changingSource = state.video.name && state.video.name !== file.name;
    if (changingSource && state.frames.length && !confirm(`Open ${file.name} and clear annotations for ${state.video.name}?`)) {
      $('#videoFileInput').value = '';
      return;
    }
    if (changingSource) state.frames = [];
    pauseMedia(); demoMode = false; els.stage.classList.remove('demo');
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    videoUrl = URL.createObjectURL(file); els.video.src = videoUrl;
    state.video.name = file.name;
    els.video.load();
    toast('Reading video metadata…');
  }

  function loadDemo() {
    pauseMedia(); demoMode = true; videoUrl = null; els.video.removeAttribute('src'); els.video.load();
    state.project.name = 'Temperate Reef Survey — Sample';
    state.project.deploymentId = 'BRUV-DEMO-01'; state.project.site = 'Emily Bay'; state.project.observer = 'Demo observer';
    state.video = { name: 'emily_bay_bruv_01.mp4', duration: 90, width: 1280, height: 720, fps: 25 };
    state.frames = [
      { id:'demo_f1', time:12.4, frameNumber:310, reviewed:true, note:'First arrival near bait arm.', savedAt:new Date().toISOString(), annotations:[
        {id:'demo_b1',speciesId:'sp_snapper',trackId:'CHRAUR-001',x:.19,y:.25,w:.14,h:.13,stage:'Adult',activity:'Passing',uncertain:false},
        {id:'demo_b2',speciesId:'sp_trevally',trackId:'PSEGE-001',x:.53,y:.39,w:.18,h:.15,stage:'Adult',activity:'Feeding',uncertain:false}]},
      { id:'demo_f2', time:42.16, frameNumber:1054, reviewed:false, note:'Peak snapper count.', savedAt:new Date().toISOString(), annotations:[
        {id:'demo_b3',speciesId:'sp_snapper',trackId:'CHRAUR-001',x:.17,y:.24,w:.15,h:.14,stage:'Adult',activity:'Feeding',uncertain:false},
        {id:'demo_b4',speciesId:'sp_snapper',trackId:'CHRAUR-002',x:.51,y:.38,w:.2,h:.16,stage:'Adult',activity:'Feeding',uncertain:false},
        {id:'demo_b5',speciesId:'sp_snapper',trackId:'CHRAUR-003',x:.37,y:.59,w:.13,h:.12,stage:'Juvenile',activity:'Passing',uncertain:false},
        {id:'demo_b6',speciesId:'sp_trevally',trackId:'PSEGE-001',x:.7,y:.17,w:.12,h:.11,stage:'Adult',activity:'Schooling',uncertain:true}]},
      { id:'demo_f3', time:68.0, frameNumber:1700, reviewed:true, note:'Wrasse passing over reef.', savedAt:new Date().toISOString(), annotations:[
        {id:'demo_b7',speciesId:'sp_wrasse',trackId:'NOTTET-001',x:.38,y:.58,w:.12,h:.11,stage:'Adult',activity:'Passing',uncertain:false},
        {id:'demo_b8',speciesId:'sp_trevally',trackId:'PSEGE-002',x:.52,y:.39,w:.18,h:.15,stage:'Adult',activity:'Schooling',uncertain:false}]}
    ];
    els.stage.classList.add('ready','demo'); els.stage.classList.remove('has-video');
    els.timeline.max = state.video.duration; setTime(42.16); renderAll(); saveProject(); toast('Sample survey loaded');
  }

  function clonePreviousFrame() {
    if (!hasMedia()) return toast('Open a video first');
    const previous = [...state.frames].filter(frame => frame.frameNumber < currentKey() && frame.annotations.length).sort((a,b) => b.frameNumber-a.frameNumber)[0];
    if (!previous) return toast('No preceding annotations to clone');
    draft = structuredClone(previous.annotations).map(ann => ({ ...ann, id: uid('box') }));
    selectedId = null; syncFrame(); toast(`Cloned ${draft.length} boxes from ${formatTime(previous.time)}`);
  }

  function getMaxFrameIds() { return new Set(summaryForSpecies().filter(row => row.maxFrame).map(row => row.maxFrame.id)); }

  function renderReview() {
    const frames = state.frames; const boxes = frames.reduce((sum,f) => sum+f.annotations.length,0); const flagged = frames.reduce((sum,f) => sum+f.annotations.filter(a=>a.uncertain).length,0); const reviewed = frames.filter(f=>f.reviewed).length;
    $('#reviewStats').innerHTML = [
      ['Annotated frames', frames.length], ['Bounding boxes', boxes], ['Needs review', flagged], ['Frames reviewed', frames.length ? `${Math.round(reviewed/frames.length*100)}%` : '0%']
    ].map(([label,value]) => `<div class="stat-card"><span>${label}</span><strong>${value}</strong></div>`).join('');
    const q = $('#reviewSearch').value.trim().toLowerCase(); const maxIds = getMaxFrameIds();
    const maxBySpecies = new Map(summaryForSpecies().map(row => [row.species.id, row.maxN]));
    const shown = frames.filter(frame => {
      if (reviewFilter === 'flagged' && !frame.annotations.some(a=>a.uncertain) && frame.reviewed) return false;
      if (reviewFilter === 'maxn' && !maxIds.has(frame.id)) return false;
      const hay = `${frame.note} ${frame.annotations.map(a => `${a.trackId} ${speciesById(a.speciesId).common}`).join(' ')}`.toLowerCase(); return hay.includes(q);
    });
    $('#reviewTableBody').innerHTML = shown.length ? shown.map(frame => {
      const grouped = new Map(); frame.annotations.forEach(a => grouped.set(a.speciesId, (grouped.get(a.speciesId) || 0) + 1));
      return `<tr><td><strong>${formatTime(frame.time)}</strong></td><td>${frame.frameNumber}</td><td>${frame.annotations.length} boxes</td><td><div class="species-pills">${[...grouped].map(([id,count]) => { const sp=speciesById(id); const atMax=count===maxBySpecies.get(id); return `<span class="tiny-pill" style="border-left:3px solid ${sp.color}">${escapeHtml(sp.code)} × ${count}${atMax?' · MAXN':''}</span>`; }).join('')}</div></td><td><span class="review-status ${frame.reviewed ? 'done' : ''}">${frame.reviewed ? 'Reviewed' : 'Needs review'}</span></td><td><button class="jump-button" data-jump-frame="${frame.frameNumber}">Open →</button></td></tr>`;
    }).join('') : '<tr class="empty-row"><td colspan="6">No annotated frames match this view.</td></tr>';
    $$('[data-jump-frame]').forEach(btn => btn.addEventListener('click', () => { selectView('annotate'); setTime(Number(btn.dataset.jumpFrame)/state.video.fps); }));
  }

  function renderDataset() {
    const rows = summaryForSpecies(); const frames = state.frames.filter(f=>f.annotations.length); const boxes = frames.reduce((s,f)=>s+f.annotations.length,0); const tracks = new Set(frames.flatMap(f=>f.annotations.map(a=>a.trackId))).size; const reviewed = state.frames.length ? Math.round(state.frames.filter(f=>f.reviewed).length/state.frames.length*100) : 0; const totalMaxn = rows.reduce((s,r)=>s+r.maxN,0);
    $('#datasetBoxes').textContent = boxes; $('#datasetFrames').textContent = frames.length; $('#datasetTracks').textContent = tracks; $('#datasetSpecies').textContent = rows.length; $('#datasetMaxn').textContent = totalMaxn; $('#datasetReviewed').textContent = `${reviewed}%`;
    const readiness = Math.min(100, (state.video.name ? 15:0) + (boxes ? 25:0) + (rows.length >= 2 ? 20:0) + Math.round(reviewed*.3) + (state.project.site ? 10:0));
    $('.health-ring').textContent = `${readiness}%`; $('#datasetHealth small').textContent = readiness === 100 ? 'Ready to export' : boxes ? 'Keep reviewing labels' : 'Add boxes to begin';
    const maxBoxes = Math.max(1, ...rows.map(r=>r.boxes));
    $('#classBalance').innerHTML = rows.length ? rows.map(row => `<div class="balance-row"><div class="balance-head"><span>${escapeHtml(row.species.common)}</span><span>${row.boxes} boxes</span></div><div class="balance-track"><div class="balance-fill" style="width:${row.boxes/maxBoxes*100}%;background:${row.species.color}"></div></div></div>`).join('') : '<div class="annotation-empty"><strong>No class data yet</strong><p>Species balance appears after the first annotations.</p></div>';
    const checks = [
      [Boolean(state.video.name),'Video metadata', state.video.name || 'No source video recorded'],
      [Boolean(state.project.site && state.project.observer),'Deployment metadata', state.project.site && state.project.observer ? `${state.project.site} · ${state.project.observer}` : 'Add site and observer'],
      [boxes >= 10,'Annotation volume', boxes ? `${boxes} boxes — aim for broad coverage` : 'No boxes yet'],
      [reviewed === 100 && state.frames.length > 0,'Quality review', `${reviewed}% of annotated frames reviewed`]
    ];
    $('#qualityChecklist').innerHTML = checks.map(([ok,title,sub]) => `<div class="check-item"><span class="check-state ${ok?'':'warn'}">${ok?'✓':'!'}</span><span class="check-copy"><strong>${title}</strong><small>${escapeHtml(sub)}</small></span></div>`).join('');
  }

  function downloadBlob(name, blob) {
    const url = URL.createObjectURL(blob); const a = document.createElement('a'); a.href=url; a.download=name; document.body.appendChild(a); a.click(); a.remove(); setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  function download(name, content, type = 'application/json') { downloadBlob(name, new Blob([content], {type})); }
  function exportProject() { download(`${safeFileName(state.project.name)}.finframe.json`, JSON.stringify(state,null,2)); toast('Project file exported'); }

  function perFrameCountLines() {
    const header = ['project','deployment_id','site','video','time_seconds','timecode','frame_number','species_code','common_name','count_in_frame','running_maxn','final_maxn','is_final_maxn_frame','reviewed'];
    const usedSpeciesIds = [...new Set(state.frames.flatMap(frame => frame.annotations.map(ann => ann.speciesId)))];
    const finalSummary = new Map(summaryForSpecies().map(row => [row.species.id, row]));
    const running = new Map(); const lines = [header];
    [...state.frames].sort((a,b)=>a.frameNumber-b.frameNumber).forEach(frame => {
      const counts = new Map(); frame.annotations.forEach(ann=>counts.set(ann.speciesId,(counts.get(ann.speciesId)||0)+1));
      usedSpeciesIds.forEach(speciesId => {
        const sp=speciesById(speciesId), count=counts.get(speciesId)||0, summary=finalSummary.get(speciesId);
        running.set(speciesId,Math.max(running.get(speciesId)||0,count));
        lines.push([state.project.name,state.project.deploymentId,state.project.site,state.video.name,frame.time.toFixed(3),formatTime(frame.time),frame.frameNumber,sp.code,sp.common,count,running.get(speciesId),summary?.maxN||0,Boolean(count && count===summary?.maxN),frame.reviewed]);
      });
    });
    return lines;
  }

  function exportFrameCsv() {
    download(`${safeFileName(state.project.name)}_per_frame_counts.csv`,csvText(perFrameCountLines()),'text/csv');
    toast('Per-frame counts exported');
  }

  function updateExportProgress(value,title,detail) {
    $('#exportProgressBar').value=value; $('#exportProgressValue').textContent=`${Math.round(value)}%`;
    if(title)$('#exportProgressTitle').textContent=title; if(detail)$('#exportProgressDetail').textContent=detail;
  }
  async function seekForExtraction(time) {
    const target=Math.max(0,Math.min(Math.max(0,els.video.duration-.001),time));
    if(Math.abs(els.video.currentTime-target)<.0005&&els.video.readyState>=2)return;
    await new Promise((resolve,reject)=>{
      const timeout=setTimeout(()=>{cleanup();reject(new Error('Timed out while seeking the source video.'));},15000);
      const cleanup=()=>{clearTimeout(timeout);els.video.removeEventListener('seeked',done);els.video.removeEventListener('error',fail);};
      const done=()=>{cleanup();resolve();}; const fail=()=>{cleanup();reject(new Error('The source video could not be decoded at an annotated frame.'));};
      els.video.addEventListener('seeked',done);els.video.addEventListener('error',fail);els.video.currentTime=target;
    });
  }
  async function extractFrameJpeg(frame) {
    await seekForExtraction(frame.time);
    const canvas=document.createElement('canvas');canvas.width=state.video.width;canvas.height=state.video.height;
    const ctx=canvas.getContext('2d',{alpha:false});ctx.drawImage(els.video,0,0,canvas.width,canvas.height);
    const blob=await new Promise(resolve=>canvas.toBlob(resolve,'image/jpeg',.94));
    if(!blob)throw new Error('The browser could not encode an extracted frame.');
    return new Uint8Array(await blob.arrayBuffer());
  }
  function extractedImageName(frame) {
    const videoBase=(state.video.name||'video').replace(/\.[^.]+$/,'');
    return `${safeFileName(videoBase)}_frame_${String(frame.frameNumber).padStart(8,'0')}.jpg`;
  }

  async function exportTrainingDataset(format) {
    const frames=state.frames.filter(frame=>frame.annotations.length).sort((a,b)=>a.frameNumber-b.frameNumber);
    if(!frames.length){toast('Add at least one labelled frame first');return;}
    if(demoMode||!videoUrl||!els.video.videoWidth){selectView('annotate');toast('Reopen the original video before extracting its labelled frames');return;}
    if(frames.length>500&&!confirm(`This will extract ${frames.length} full-resolution frames in browser memory. Continue?`))return;
    pauseMedia(); const originalTime=currentTime, dialog=$('#exportProgressDialog'), zip=new ZipStore();
    dialog.showModal();updateExportProgress(0,'Extracting labelled frames',`0 of ${frames.length} frames`);
    try {
      const imageRecords=[];
      for(let index=0;index<frames.length;index+=1){
        const frame=frames[index],fileName=extractedImageName(frame),bytes=await extractFrameJpeg(frame);
        zip.add(`images/${fileName}`,bytes);imageRecords.push({frame,fileName,imageId:index+1});
        if(format==='yolo'){
          const classMap=new Map(state.species.map((sp,i)=>[sp.id,i]));
          const labels=frame.annotations.map(a=>yoloLineFromNormalized(classMap.get(a.speciesId),a)).join('\n');
          zip.add(`labels/${fileName.replace(/\.jpg$/,'.txt')}`,labels);
        }
        updateExportProgress((index+1)/frames.length*88,'Extracting labelled frames',`${index+1} of ${frames.length} frames`);
      }
      if(format==='coco'){
        const categories=state.species.map((sp,i)=>({id:i+1,name:sp.common,supercategory:'fish',scientific_name:sp.scientific,code:sp.code}));
        const categoryMap=new Map(state.species.map((sp,i)=>[sp.id,i+1]));let annotationId=1;
        const images=imageRecords.map(({frame,fileName,imageId})=>({id:imageId,file_name:`images/${fileName}`,width:state.video.width,height:state.video.height,video_file:state.video.name,frame_number:frame.frameNumber,timestamp_seconds:Number(frame.time.toFixed(3)),deployment_id:state.project.deploymentId}));
        const annotations=imageRecords.flatMap(({frame,imageId})=>frame.annotations.map(ann=>{const bbox=cocoBoxFromNormalized(ann,state.video.width,state.video.height);return{id:annotationId++,image_id:imageId,category_id:categoryMap.get(ann.speciesId),bbox,area:Number((bbox[2]*bbox[3]).toFixed(2)),iscrowd:0,track_id:ann.trackId,attributes:{stage:ann.stage,activity:ann.activity,uncertain:ann.uncertain}};}));
        zip.add('annotations/instances.json',JSON.stringify({info:{description:state.project.name,version:'1.0',created:new Date().toISOString(),source:'FinFrame',deployment_id:state.project.deploymentId},images,annotations,categories},null,2));
      } else {
        const yaml=['# FinFrame YOLO dataset','# Keep complete deployments together when creating train/val/test splits.','path: .','train: images','names:',...state.species.map((sp,i)=>`  ${i}: ${JSON.stringify(sp.common)}`)].join('\n');
        zip.add('data.yaml',yaml);zip.add('classes.txt',state.species.map(sp=>sp.common).join('\n'));
      }
      zip.add('metadata/project.finframe.json',JSON.stringify(state,null,2));
      zip.add('metadata/per_frame_counts.csv',csvText(perFrameCountLines()));
      zip.add('README.txt',`FinFrame ${format.toUpperCase()} export\n\nEach JPEG is the exact annotated video frame. Bounding boxes use the uncropped source dimensions ${state.video.width}x${state.video.height}.\n\nDo not randomly split neighbouring frames from this deployment across training and validation sets; split by deployment/video to prevent data leakage.\n`);
      updateExportProgress(94,'Packaging dataset','Building ZIP archive…');
      const blob=new Blob([zip.build()],{type:'application/zip'});updateExportProgress(100,'Dataset ready',`${frames.length} images · ${(blob.size/1024/1024).toFixed(1)} MB`);
      downloadBlob(`${safeFileName(state.project.name)}_${format}_dataset.zip`,blob);
      setTimeout(()=>dialog.close(),650);toast(`${format.toUpperCase()} dataset exported with ${frames.length} images`);
    } catch(error) {
      dialog.close();toast(error.message||'Dataset extraction failed');
    } finally {
      try{await seekForExtraction(originalTime);}catch(_){/* preserve export result */}
      currentTime=originalTime;els.timeline.value=originalTime;loadFrameAtTime(true);
    }
  }

  function exportCoco() { return exportTrainingDataset('coco'); }
  function exportYolo() { return exportTrainingDataset('yolo'); }
  function exportCsv() {
    const header=['project','deployment_id','site','observer','video','time_seconds','timecode','frame_number','species_code','common_name','scientific_name','count_in_frame','is_maxn_frame','maxn','first_arrival_seconds','mean_count','stage','track_id','uncertain','x','y','width','height','reviewed','note'];
    const summary = new Map(summaryForSpecies().map(row=>[row.species.id,row])); const lines=[header];
    state.frames.forEach(frame => { const counts=new Map(); frame.annotations.forEach(a=>counts.set(a.speciesId,(counts.get(a.speciesId)||0)+1)); frame.annotations.forEach(a=>{ const sp=speciesById(a.speciesId), s=summary.get(a.speciesId); lines.push([state.project.name,state.project.deploymentId,state.project.site,state.project.observer,state.video.name,frame.time.toFixed(3),formatTime(frame.time),frame.frameNumber,sp.code,sp.common,sp.scientific,counts.get(a.speciesId),s?.maxFrame?.id===frame.id,s?.maxN??0,s?.first?.toFixed(3)??'',s?.mean?.toFixed(3)??'',a.stage,a.trackId,a.uncertain,a.x,a.y,a.w,a.h,frame.reviewed,frame.note]); }); });
    download(`${safeFileName(state.project.name)}_observations.csv`,csvText(lines),'text/csv'); toast('Observation CSV exported');
  }

  function renderAll() { renderProjectInfo(); renderSpecies(); renderFrame(); renderMaxN(); }

  $$('.nav-item').forEach(btn => btn.addEventListener('click', () => selectView(btn.dataset.view)));
  $('#backToAnnotateBtn').addEventListener('click', () => selectView('annotate'));
  els.speciesSearch.addEventListener('input', renderSpecies);
  $$('[data-tool]').forEach(btn => btn.addEventListener('click', () => setTool(btn.dataset.tool)));
  $$('.inspector-tab').forEach(btn => btn.addEventListener('click', () => {
    $$('.inspector-tab').forEach(b=>b.classList.toggle('is-active',b===btn));
    $$('.inspector-pane').forEach(p=>p.classList.remove('is-visible'));
    $(`#${btn.dataset.inspector}Inspector`).classList.add('is-visible');
  }));
  $('#videoFileInput').addEventListener('change', event => loadVideo(event.target.files[0]));
  $('#demoBtn').addEventListener('click', loadDemo); $('#stageDemoBtn').addEventListener('click', loadDemo);
  els.video.addEventListener('loadedmetadata', () => {
    state.video.duration=els.video.duration; state.video.width=els.video.videoWidth; state.video.height=els.video.videoHeight; currentTime=0; currentFrameKey=null;
    els.timeline.max=state.video.duration; els.stage.classList.add('ready','has-video'); renderAll(); resizeCanvas(); saveProject(); toast('Video ready to annotate');
  });
  els.video.addEventListener('timeupdate', () => { currentTime=els.video.currentTime; els.timeline.value=currentTime; loadFrameAtTime(); updateTimeLabels(); });
  els.video.addEventListener('play', () => { els.playBtn.textContent='Ⅱ'; els.playBtn.setAttribute('aria-label','Pause'); });
  els.video.addEventListener('pause', () => { els.playBtn.textContent='▶'; els.playBtn.setAttribute('aria-label','Play'); });
  els.video.addEventListener('ended', pauseMedia);
  els.timeline.addEventListener('input', () => { pauseMedia(); setTime(Number(els.timeline.value)); });
  els.playBtn.addEventListener('click', togglePlayback); $('#jumpBackBtn').addEventListener('click',()=>setTime(currentTime-5)); $('#jumpForwardBtn').addEventListener('click',()=>setTime(currentTime+5)); $('#prevFrameBtn').addEventListener('click',()=>stepFrame(-1)); $('#nextFrameBtn').addEventListener('click',()=>stepFrame(1));
  els.fpsInput.addEventListener('change',()=>{ state.video.fps=Math.max(1,Math.min(120,Number(els.fpsInput.value)||25)); currentFrameKey=null; loadFrameAtTime(true); saveProject(); });
  els.speedSelect.addEventListener('change',()=>{ els.video.playbackRate=Number(els.speedSelect.value); });
  els.canvas.addEventListener('pointerdown',onPointerDown); els.canvas.addEventListener('pointermove',onPointerMove); els.canvas.addEventListener('pointerup',onPointerUp); els.canvas.addEventListener('pointercancel',()=>{pointerAction=null;drawCanvas();});
  $('#cloneBtn').addEventListener('click',clonePreviousFrame);
  $('#clearFrameBtn').addEventListener('click',()=>{ if(!draft.length)return; if(confirm('Remove every box from this frame?')){draft=[];selectedId=null;syncFrame();} });
  els.frameNote.addEventListener('change',()=>syncFrame({createEmpty:Boolean(els.frameNote.value.trim())}));
  $('#markReviewedBtn').addEventListener('click',()=>{ let frame=getFrame(); if(!frame){syncFrame({createEmpty:true});frame=getFrame();} frame.reviewed=!frame.reviewed;saveProject();renderFrame();toast(frame.reviewed?'Frame marked reviewed':'Review status removed'); });
  $('#addSpeciesBtn').addEventListener('click',()=>$('#speciesDialog').showModal());
  $('#speciesForm').addEventListener('submit',event=>{ const submitter=event.submitter;if(submitter?.value==='cancel')return;event.preventDefault();const data=new FormData(event.currentTarget);const code=String(data.get('code')).trim().toUpperCase();const sp={id:uid('sp'),common:String(data.get('commonName')).trim(),scientific:String(data.get('scientificName')).trim(),code,color:String(data.get('color'))};state.species.push(sp);state.activeSpeciesId=sp.id;saveProject();renderSpecies();event.currentTarget.reset();$('#speciesDialog').close();toast(`${sp.common} added`); });
  $('#detailsBtn').addEventListener('click',()=>{ const form=$('#detailsForm'); ['projectName','deploymentId','site','observer','date','depth','notes'].forEach(name=>form.elements[name].value=name==='projectName'?state.project.name:(state.project[name]??'')); $('#detailsDialog').showModal(); });
  $('#projectTitleBtn').addEventListener('click',()=>$('#detailsBtn').click());
  $('#detailsForm').addEventListener('submit',event=>{ if(event.submitter?.value==='cancel')return;event.preventDefault();const data=Object.fromEntries(new FormData(event.currentTarget));state.project={...state.project,name:data.projectName,...data};delete state.project.projectName;saveProject();renderProjectInfo();$('#detailsDialog').close();toast('Deployment details saved'); });
  $('#exportProjectBtn').addEventListener('click',exportProject);
  $('#importProjectBtn').addEventListener('click',()=>$('#projectFileInput').click());
  $('#projectFileInput').addEventListener('change',async event=>{ try{const incoming=JSON.parse(await event.target.files[0].text());if(!incoming.schemaVersion||!Array.isArray(incoming.frames)||!Array.isArray(incoming.species))throw new Error();state=incoming;demoMode=false;videoUrl=null;currentTime=0;currentFrameKey=null;els.stage.classList.remove('ready','has-video','demo');renderAll();saveProject();toast('Project imported — reopen its video to continue');}catch(_){toast('That file is not a valid FinFrame project');}event.target.value=''; });
  $$('.table-tab').forEach(btn=>btn.addEventListener('click',()=>{reviewFilter=btn.dataset.reviewFilter;$$('.table-tab').forEach(b=>b.classList.toggle('is-active',b===btn));renderReview();}));
  $('#reviewSearch').addEventListener('input',renderReview); $('#reviewAllBtn').addEventListener('click',()=>{state.frames.forEach(f=>f.reviewed=true);saveProject();renderReview();toast('All annotated frames marked reviewed');});
  $$('[data-export]').forEach(btn=>btn.addEventListener('click',()=>({coco:exportCoco,yolo:exportYolo,csv:exportCsv,framecsv:exportFrameCsv,project:exportProject}[btn.dataset.export]())));
  window.addEventListener('resize',resizeCanvas); new ResizeObserver(resizeCanvas).observe(els.stage);
  document.addEventListener('keydown',event=>{ if(/INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName))return; if(event.key===' '){event.preventDefault();togglePlayback();} if(event.key==='ArrowLeft'){event.preventDefault();stepFrame(-1);} if(event.key==='ArrowRight'){event.preventDefault();stepFrame(1);} if(event.key.toLowerCase()==='b')setTool('draw'); if(event.key.toLowerCase()==='v')setTool('select'); if((event.key==='Delete'||event.key==='Backspace')&&selectedId)deleteAnnotation(selectedId); });

  if (state.video.name) {
    $('#projectSubtitle').textContent = `${state.video.name} · reopen video to resume`;
  }
  els.timeline.max = state.video.duration || 100;
  renderAll(); resizeCanvas();
})();
