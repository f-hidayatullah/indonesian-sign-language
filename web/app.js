const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const statusEl = document.getElementById('status');
const phraseEl = document.getElementById('phrase');
const textOutputEl = document.getElementById('textOutput');
const audioStatusEl = document.getElementById('audioStatus');
const apiUrlEl = document.getElementById('apiUrl');
const pipelineSelectEl = document.getElementById('pipelineSelect');
const inputQualityEl = document.getElementById('inputQuality');
const detectionBoxEl = document.getElementById('detectionBox');
const targetLetterEl = document.getElementById('targetLetter');
const recordEvalBtn = document.getElementById('recordEvalBtn');
const downloadEvalBtn = document.getElementById('downloadEvalBtn');
const resetEvalBtn = document.getElementById('resetEvalBtn');
const evalSummaryEl = document.getElementById('evalSummary');
const evalRowsEl = document.getElementById('evalRows');
const startBtn = document.getElementById('startBtn');
const switchBtn = document.getElementById('switchBtn');
const stopBtn = document.getElementById('stopBtn');

let stream = null;
let timer = null;
let lastSpoken = '';
let lastSpokenAt = 0;
let currentFacingMode = 'environment';
let selectedVoice = null;
let predictionHistory = [];
let stablePhrase = '';
let emptyFrameCount = 0;
let latestInputQuality = '-';
let evaluationRows = [];

const SEND_INTERVAL_MS = 600;
const SPEAK_COOLDOWN_MS = 2500;
const STABILITY_WINDOW = 4;
const STABILITY_MIN_VOTES = 3;
const EMPTY_RESET_FRAMES = 2;
const MIN_BRIGHTNESS = 45;
const MIN_HAND_AREA_RATIO = 0.04;
const MAX_HAND_AREA_RATIO = 0.72;
const EDGE_MARGIN_RATIO = 0.04;
const LETTER_OPTIONS = [
  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
  'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
  'U', 'V', 'W', 'X', 'Y', 'O'
];
const hasSpeechSynthesis =
  typeof window !== 'undefined' &&
  'speechSynthesis' in window &&
  typeof window.SpeechSynthesisUtterance !== 'undefined';

function setStatus(text) {
  statusEl.textContent = text;
}

function setInputQuality(text, level = '') {
  latestInputQuality = text;
  const valueEl = inputQualityEl.querySelector('strong');
  if (valueEl) valueEl.textContent = text;
  inputQualityEl.className = `meta-item camera-quality ${level}`.trim();
}

function hideDetectionBox() {
  detectionBoxEl.style.display = 'none';
  detectionBoxEl.dataset.label = '';
}

function selectedPipeline() {
  return pipelineSelectEl.value;
}

function resetPredictionState() {
  predictionHistory = [];
  stablePhrase = '';
  emptyFrameCount = 0;
  lastSpoken = '';
  lastSpokenAt = 0;
  audioStatusEl.textContent = 'menunggu gestur tangan';
}

function initEvaluationControls() {
  targetLetterEl.innerHTML = LETTER_OPTIONS
    .map((letter) => `<option value="${letter}">${letter}</option>`)
    .join('');
}

function updateEvaluationSummary() {
  const total = evaluationRows.length;
  const correct = evaluationRows.filter((row) => row.correct).length;
  const noPred = evaluationRows.filter((row) => row.output === '-').length;
  const accuracy = total ? ((correct / total) * 100).toFixed(1) : '0.0';
  evalSummaryEl.textContent = `Evaluasi: ${total} percobaan | benar ${correct} | NO_PRED ${noPred} | akurasi ${accuracy}%`;
}

function renderEvaluationRows() {
  evalRowsEl.innerHTML = evaluationRows
    .slice(-20)
    .map((row) => `
      <tr>
        <td>${row.trial}</td>
        <td>${row.pipeline}</td>
        <td>${row.camera}</td>
        <td>${row.target}</td>
        <td>${row.output}</td>
        <td>${row.correct ? 'benar' : 'salah'}</td>
        <td>${row.inputQuality}</td>
      </tr>
    `)
    .join('');
}

function recordEvaluationTrial() {
  const target = targetLetterEl.value;
  const output = (phraseEl.textContent || '-').trim() || '-';
  const row = {
    trial: evaluationRows.length + 1,
    timestamp: new Date().toISOString(),
    pipeline: selectedPipeline(),
    camera: currentFacingMode,
    target,
    output,
    correct: output === target,
    status: statusEl.textContent,
    inputQuality: latestInputQuality,
  };
  evaluationRows.push(row);
  renderEvaluationRows();
  updateEvaluationSummary();
}

function csvEscape(value) {
  const text = String(value ?? '');
  return `"${text.replace(/"/g, '""')}"`;
}

function downloadEvaluationCsv() {
  const headers = ['trial', 'timestamp', 'pipeline', 'camera', 'target', 'output', 'correct', 'status', 'inputQuality'];
  const lines = [
    headers.join(','),
    ...evaluationRows.map((row) => headers.map((key) => csvEscape(row[key])).join(',')),
  ];
  const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `evaluasi-kamera-nyata-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.csv`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function resetEvaluation() {
  evaluationRows = [];
  renderEvaluationRows();
  updateEvaluationSummary();
}

function estimateBrightness(width, height) {
  const sampleWidth = Math.max(1, Math.floor(width / 8));
  const sampleHeight = Math.max(1, Math.floor(height / 8));
  const sampleCanvas = document.createElement('canvas');
  sampleCanvas.width = sampleWidth;
  sampleCanvas.height = sampleHeight;
  const sampleCtx = sampleCanvas.getContext('2d');
  sampleCtx.drawImage(canvas, 0, 0, sampleWidth, sampleHeight);
  const data = sampleCtx.getImageData(0, 0, sampleWidth, sampleHeight).data;
  let total = 0;
  for (let i = 0; i < data.length; i += 4) {
    total += 0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2];
  }
  return total / (data.length / 4);
}

function assessInputQuality(predictions, width, height, brightness) {
  if (brightness < MIN_BRIGHTNESS) {
    return { text: 'cahaya rendah', level: 'bad' };
  }

  if (!predictions.length) {
    return { text: 'tangan belum terdeteksi', level: 'warn' };
  }

  const first = predictions[0];
  if (!Array.isArray(first.bbox) || first.bbox.length !== 4) {
    return { text: 'deteksi belum lengkap', level: 'warn' };
  }

  const [x1, y1, x2, y2] = first.bbox;
  const areaRatio = Math.max(0, (x2 - x1) * (y2 - y1)) / Math.max(1, width * height);
  if (areaRatio < MIN_HAND_AREA_RATIO) {
    return { text: 'tangan terlalu jauh', level: 'warn' };
  }
  if (areaRatio > MAX_HAND_AREA_RATIO) {
    return { text: 'tangan terlalu dekat', level: 'warn' };
  }

  const edgeX = width * EDGE_MARGIN_RATIO;
  const edgeY = height * EDGE_MARGIN_RATIO;
  if (x1 <= edgeX || y1 <= edgeY || x2 >= width - edgeX || y2 >= height - edgeY) {
    return { text: 'tangan terlalu pinggir', level: 'warn' };
  }

  const confidence = first.yolo_conf ?? first.rf_conf ?? null;
  if (confidence !== null && confidence < 0.3) {
    return { text: 'prediksi belum yakin', level: 'warn' };
  }

  return { text: 'siap', level: 'good' };
}

function renderDetectionBox(predictions, frameWidth, frameHeight) {
  if (!predictions.length) {
    hideDetectionBox();
    return;
  }

  const first = predictions[0];
  if (!Array.isArray(first.bbox) || first.bbox.length !== 4) {
    hideDetectionBox();
    return;
  }

  const [x1, y1, x2, y2] = first.bbox;
  if (x2 <= x1 || y2 <= y1) {
    hideDetectionBox();
    return;
  }

  const rect = video.getBoundingClientRect();
  const scale = Math.max(rect.width / frameWidth, rect.height / frameHeight);
  const drawnWidth = frameWidth * scale;
  const drawnHeight = frameHeight * scale;
  const offsetX = (rect.width - drawnWidth) / 2;
  const offsetY = (rect.height - drawnHeight) / 2;

  const left = offsetX + x1 * scale;
  const top = offsetY + y1 * scale;
  const width = (x2 - x1) * scale;
  const height = (y2 - y1) * scale;

  detectionBoxEl.style.display = 'block';
  detectionBoxEl.style.left = `${Math.max(0, left)}px`;
  detectionBoxEl.style.top = `${Math.max(0, top)}px`;
  detectionBoxEl.style.width = `${Math.min(rect.width - Math.max(0, left), width)}px`;
  detectionBoxEl.style.height = `${Math.min(rect.height - Math.max(0, top), height)}px`;
  detectionBoxEl.dataset.label = first.label || '';
}

function getStablePhrase(phrase) {
  const normalized = (phrase || '').trim();
  predictionHistory.push(normalized);

  if (predictionHistory.length > STABILITY_WINDOW) {
    predictionHistory.shift();
  }

  if (!normalized) {
    emptyFrameCount += 1;
    if (emptyFrameCount >= EMPTY_RESET_FRAMES) {
      stablePhrase = '';
      predictionHistory = [];
    }
    return stablePhrase;
  }

  emptyFrameCount = 0;

  const votes = predictionHistory.reduce((counts, item) => {
    if (item) counts[item] = (counts[item] || 0) + 1;
    return counts;
  }, {});

  const winner = Object.entries(votes).sort((a, b) => b[1] - a[1])[0];
  if (winner && winner[1] >= STABILITY_MIN_VOTES) {
    stablePhrase = winner[0];
  }

  return stablePhrase;
}

function selectVoice() {
  if (!hasSpeechSynthesis) return;

  const voices = window.speechSynthesis.getVoices();
  selectedVoice =
    voices.find((voice) => voice.lang && voice.lang.toLowerCase().startsWith('id')) ||
    voices.find((voice) => voice.lang && voice.lang.toLowerCase().startsWith('en')) ||
    null;
}

function speakId(text) {
  if (!hasSpeechSynthesis) return;

  const now = Date.now();
  if (!text) return;
  if (text === lastSpoken && now - lastSpokenAt < SPEAK_COOLDOWN_MS) return;

  const utter = new window.SpeechSynthesisUtterance(text);
  utter.lang = selectedVoice?.lang || 'id-ID';
  utter.volume = 1;
  utter.rate = 0.9;
  utter.pitch = 1;

  if (selectedVoice) {
    utter.voice = selectedVoice;
  }

  if (window.speechSynthesis.speaking) {
    window.speechSynthesis.cancel();
  }

  window.speechSynthesis.speak(utter);
  lastSpoken = text;
  lastSpokenAt = now;
}

async function sendFrame() {
  if (!stream) return;

  const w = video.videoWidth;
  const h = video.videoHeight;
  if (!w || !h) return;

  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, w, h);
  const brightness = estimateBrightness(w, h);

  const dataUrl = canvas.toDataURL('image/jpeg', 0.8);

  try {
    const res = await fetch(apiUrlEl.value.trim(), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_base64: dataUrl,
        pipeline: selectedPipeline(),
        max_hands: 2
      })
    });

    if (!res.ok) {
      setStatus(`error ${res.status}`);
      return;
    }

    const json = await res.json();
    const predictions = Array.isArray(json.predictions) ? json.predictions : [];
    renderDetectionBox(predictions, w, h);
    const inputQuality = assessInputQuality(predictions, w, h, brightness);
    setInputQuality(inputQuality.text, inputQuality.level);

    const rawPhrase = (json.phrase || '').trim();
    const phrase = getStablePhrase(rawPhrase);
    phraseEl.textContent = phrase || '-';
    textOutputEl.textContent = phrase || '-';
    audioStatusEl.textContent = phrase ? 'aktif' : 'menunggu gestur tangan';
    setStatus(
      hasSpeechSynthesis
        ? `running (${selectedPipeline()}, ${phrase ? 'stable' : 'stabilizing'})`
        : `running (${selectedPipeline()}, ${phrase ? 'stable' : 'stabilizing'}) - audio tidak didukung browser ini`
    );
    if (phrase) speakId(phrase);
  } catch (err) {
    setStatus(`request failed: ${err.message}`);
  }
}

async function startCamera() {
  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setStatus('camera API tidak tersedia. Buka lewat HTTPS dan browser utama (bukan in-app browser).');
      return;
    }
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: currentFacingMode },
      audio: false
    });
    video.srcObject = stream;
    await video.play();
    resetPredictionState();
    phraseEl.textContent = '-';
    textOutputEl.textContent = '-';
    hideDetectionBox();
    setInputQuality('menunggu kamera', 'warn');
    pipelineSelectEl.disabled = true;
    setStatus(
      hasSpeechSynthesis
        ? `camera on (${currentFacingMode}, ${selectedPipeline()})`
        : `camera on (${currentFacingMode}, ${selectedPipeline()}) - audio tidak didukung browser ini`
    );

    if (timer) clearInterval(timer);
    timer = setInterval(sendFrame, SEND_INTERVAL_MS);
  } catch (err) {
    setStatus(`camera error: ${err.message}`);
  }
}

async function switchCamera() {
  currentFacingMode = currentFacingMode === 'user' ? 'environment' : 'user';
  stopCamera();
  await startCamera();
}

function stopCamera() {
  if (timer) {
    clearInterval(timer);
    timer = null;
  }
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  video.srcObject = null;
  pipelineSelectEl.disabled = false;
  resetPredictionState();
  phraseEl.textContent = '-';
  textOutputEl.textContent = '-';
  hideDetectionBox();
  setInputQuality('-', '');
  setStatus('stopped');
}

if (hasSpeechSynthesis) {
  selectVoice();
  window.speechSynthesis.onvoiceschanged = selectVoice;
}

initEvaluationControls();
updateEvaluationSummary();

startBtn.addEventListener('click', startCamera);
switchBtn.addEventListener('click', switchCamera);
stopBtn.addEventListener('click', stopCamera);
recordEvalBtn.addEventListener('click', recordEvaluationTrial);
downloadEvalBtn.addEventListener('click', downloadEvaluationCsv);
resetEvalBtn.addEventListener('click', resetEvaluation);
