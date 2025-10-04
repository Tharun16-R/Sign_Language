// Frontend: capture video, run MediaPipe Hands in browser, send sliding window features to backend /predict

// Read config from meta element injected by Flask template
const cfgEl = document.getElementById('config');
const SEQ_LEN = parseInt((cfgEl && cfgEl.dataset.seqLen) ? cfgEl.dataset.seqLen : '30', 10);
let LABELS = [];
try {
  LABELS = cfgEl && cfgEl.dataset.labels ? JSON.parse(cfgEl.dataset.labels) : [];
} catch { LABELS = []; }

const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const predEl = document.getElementById('prediction');
const fpsEl = document.getElementById('fps');

let featureBuffer = [];

function landmarksToFeatures(landmarks) {
  // Expect 21 landmarks with x,y,z
  const F = 63;
  if (!landmarks || landmarks.length !== 21) return new Array(F).fill(0);
  const arr = [];
  for (const lm of landmarks) arr.push(lm.x, lm.y, lm.z);
  return arr;
}

function appendBuffer(feat) {
  featureBuffer.push(feat);
  if (featureBuffer.length > SEQ_LEN) featureBuffer.shift();
}

async function predictIfReady() {
  if (featureBuffer.length < SEQ_LEN) return;
  const seq = featureBuffer.slice(-SEQ_LEN);
  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sequence: seq })
    });
    if (!res.ok) return;
    const json = await res.json();
    predEl.textContent = `${json.prediction}`;
  } catch (e) { /* ignore */ }
}

// Setup MediaPipe Hands
const hands = new Hands({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
});
hands.setOptions({
  maxNumHands: 1,
  modelComplexity: 1,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});
hands.onResults(onResults);

let lastT = performance.now();

function onResults(results) {
  // Draw video frame to canvas mirrored
  ctx.save();
  ctx.scale(-1, 1);
  ctx.drawImage(results.image, -canvas.width, 0, canvas.width, canvas.height);
  ctx.restore();

  let lm = null;
  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    lm = results.multiHandLandmarks[0];
    // draw landmarks
    drawConnectors(ctx, lm, HAND_CONNECTIONS, { color: '#22d3ee', lineWidth: 2 });
    drawLandmarks(ctx, lm, { color: '#a78bfa', lineWidth: 1 });
  }

  const feat = landmarksToFeatures(lm);
  appendBuffer(feat);

  const t = performance.now();
  const dt = t - lastT;
  lastT = t;
  const fps = 1000 / dt;
  fpsEl.textContent = fps.toFixed(1);

  predictIfReady();
}

async function initCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;
  await video.play();
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  const camera = new Camera(video, {
    onFrame: async () => { await hands.send({ image: video }); },
    width: video.videoWidth,
    height: video.videoHeight
  });
  camera.start();
}

initCamera();

// Text to Speech
const ttsBtn = document.getElementById('ttsBtn');
const ttsText = document.getElementById('ttsText');

ttsBtn.addEventListener('click', async () => {
  const text = ttsText.value || '';
  if (!text) return;
  const res = await fetch('/text_to_speech', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }) });
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const audio = new Audio(url);
  audio.play();
});

// Text -> Sign demo
const text2signBtn = document.getElementById('text2signBtn');
const signSequence = document.getElementById('signSequence');

text2signBtn.addEventListener('click', async () => {
  const text = ttsText.value || '';
  const res = await fetch(`/text_to_sign?text=${encodeURIComponent(text)}`);
  const json = await res.json();
  signSequence.innerHTML = '';
  for (const item of json.sequence) {
    const card = document.createElement('div');
    card.className = 'sign-card';
    card.textContent = item.char;
    if (item.url) {
      const vid = document.createElement('video');
      vid.src = item.url;
      vid.width = 80;
      vid.height = 60;
      vid.muted = true;
      vid.autoplay = true;
      vid.loop = true;
      card.appendChild(document.createElement('br'));
      card.appendChild(vid);
    }
    signSequence.appendChild(card);
  }
});
