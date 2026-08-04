// AICC Banking Assistant - customer call screen (mobile-first).
// Vanilla JS. Continuous hands-free flow: tap Start Call, talk, the phone
// records each utterance on silence, sends it to /api/voice-turn, plays the
// agent reply, then resumes listening. Mirrors the developer UI's VAD but with
// a simple call-style interface and extra mobile-browser handling.

// --- VAD tuning (same conservative defaults as the developer UI) ------------
const VAD_CHECK_INTERVAL_MS = 50;   // how often we sample mic volume
const SPEECH_START_CHECKS = 3;      // consecutive loud checks (~150ms) = speech started
const SILENCE_MS = 900;             // silence this long ends the utterance
const VAD_THRESHOLD = 0.02;         // RMS above this counts as speech (0..1)
const MIN_UTTERANCE_MS = 400;       // ignore blips shorter than this

// --- DOM ---
const startBtn = document.getElementById("start-call");
const endBtn = document.getElementById("end-call");
const tapToPlayBtn = document.getElementById("tap-to-play");
const statusEl = document.getElementById("status");
const statusText = document.getElementById("status-text");
const liveInterim = document.getElementById("live-interim");
const liveYou = document.getElementById("live-you");
const liveAgent = document.getElementById("live-agent");
const historyEl = document.getElementById("history");
const agentAudio = document.getElementById("agent-audio");

const sttSelect = document.getElementById("stt-provider");
const ttsSelect = document.getElementById("tts-provider");
const routerSelect = document.getElementById("router-mode");
const promptSelect = document.getElementById("prompt-mode");
const modelInput = document.getElementById("llm-model");
const ttsOutputSelect = document.getElementById("tts-output-mode");
const dbgRoute = document.getElementById("dbg-route");
const dbgSource = document.getElementById("dbg-source");
const dbgLatency = document.getElementById("dbg-latency");

// --- Session tracking ---
// One session id per call, sent with every API/socket request so the server can
// group all logs for this call under output/logs/sessions/<session_id>/.
// A fresh id is created when a call starts and cleared when it ends.
let sessionId = null;

function newSessionId() {
  const random = Math.random().toString(16).slice(2, 10) + Date.now().toString(16).slice(-4);
  return "s_" + random.slice(0, 12);
}

function startSession() {
  sessionId = newSessionId();
  updateSessionDisplay();
  return sessionId;
}

function getSessionId() {
  if (!sessionId) startSession();
  return sessionId;
}

function endSession() {
  sessionId = null;
  updateSessionDisplay();
}

function updateSessionDisplay() {
  const el = document.getElementById("dbg-session");
  if (el) el.textContent = sessionId || "—";
}

// --- Call state ---
let callActive = false;
let state = "ready";
let micStream = null;
let audioContext = null;
let analyser = null;
let vadTimer = null;
let recorder = null;
let chunks = [];
let loudCount = 0;
let lastLoudTime = 0;
let utteranceStart = 0;
let recorderMime = "";

const STATUS_LABEL = {
  ready: "Ready",
  connecting: "Connecting…",
  listening: "Listening…",
  user_speaking: "You're speaking…",
  processing: "Processing…",
  agent_speaking: "Agent speaking…",
  ended: "Call ended",
  error: "Error",
  // streaming-STT-specific states (used by customer_streaming.js)
  streaming_listening: "Streaming STT connected — listening…",
  receiving_transcript: "Hearing you…",
  processing_final_transcript: "Processing…",
};

function setState(next, messageOverride) {
  state = next;
  statusEl.className = "status status-" + next;
  statusText.textContent = messageOverride || STATUS_LABEL[next] || next;
}

// --- Mobile audio: pick a MIME type this browser's MediaRecorder supports ---
function pickSupportedMime() {
  const options = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/mp4",
    "audio/aac",
  ];
  if (typeof MediaRecorder === "undefined" || !MediaRecorder.isTypeSupported) {
    return ""; // let the browser choose its default
  }
  for (const type of options) {
    if (MediaRecorder.isTypeSupported(type)) return type;
  }
  return "";
}

function fileExtForMime(mime) {
  if (mime.includes("mp4") || mime.includes("aac")) return "m4a";
  return "webm";
}

// --- Mobile audio: unlock the <audio> element inside the Start Call gesture,
// so later programmatic play() calls (after a network round-trip) are allowed.
function buildSilentWavUrl() {
  const sampleRate = 8000;
  const samples = 400; // ~0.05s of silence
  const buffer = new ArrayBuffer(44 + samples * 2);
  const view = new DataView(buffer);
  const writeStr = (offset, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };
  writeStr(0, "RIFF"); view.setUint32(4, 36 + samples * 2, true); writeStr(8, "WAVE");
  writeStr(12, "fmt "); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
  view.setUint16(22, 1, true); view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  writeStr(36, "data"); view.setUint32(40, samples * 2, true);
  // sample bytes are already zero = silence
  return URL.createObjectURL(new Blob([view], { type: "audio/wav" }));
}

function unlockAudioPlayback() {
  agentAudio.playsInline = true;
  agentAudio.setAttribute("playsinline", "");
  try {
    agentAudio.src = buildSilentWavUrl();
    const p = agentAudio.play();
    if (p && p.then) {
      p.then(() => { agentAudio.pause(); agentAudio.currentTime = 0; }).catch(() => {});
    }
  } catch (e) {
    // Non-fatal: if unlock fails we fall back to the "Tap to play" button later.
  }
}

// --- Start / End call ---
async function startCall() {
  startBtn.disabled = true;
  tapToPlayBtn.classList.add("hidden");
  startSession(); // fresh session id for this call
  setState("connecting");

  unlockAudioPlayback(); // must happen inside the click gesture

  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    setState("error", "Microphone permission was denied.");
    startBtn.disabled = false;
    return;
  }

  audioContext = new (window.AudioContext || window.webkitAudioContext)();
  if (audioContext.state === "suspended") {
    try { await audioContext.resume(); } catch (e) { /* ignore */ }
  }
  const source = audioContext.createMediaStreamSource(micStream);
  analyser = audioContext.createAnalyser();
  analyser.fftSize = 2048;
  source.connect(analyser);

  recorderMime = pickSupportedMime();

  callActive = true;
  loudCount = 0;
  endBtn.disabled = false;
  setState("listening");
  vadTimer = setInterval(vadTick, VAD_CHECK_INTERVAL_MS);
}

function endCall(finalState) {
  callActive = false;
  if (vadTimer) { clearInterval(vadTimer); vadTimer = null; }
  if (recorder && recorder.state === "recording") {
    recorder.ondataavailable = null;
    recorder.onstop = null;
    try { recorder.stop(); } catch (e) { /* ignore */ }
  }
  recorder = null;
  if (micStream) { micStream.getTracks().forEach((t) => t.stop()); micStream = null; }
  if (audioContext) { try { audioContext.close(); } catch (e) {} audioContext = null; analyser = null; }
  try { agentAudio.pause(); } catch (e) {}

  startBtn.disabled = false;
  endBtn.disabled = true;
  liveYou.textContent = "";
  liveAgent.textContent = "";
  endSession(); // next call gets a fresh session id
  setState(finalState || "ended");
}

// --- VAD: measure mic volume and detect utterance start/end ---
function currentVolume() {
  const data = new Uint8Array(analyser.fftSize);
  analyser.getByteTimeDomainData(data);
  let sum = 0;
  for (let i = 0; i < data.length; i++) {
    const v = (data[i] - 128) / 128;
    sum += v * v;
  }
  return Math.sqrt(sum / data.length);
}

// Feedback-loop guard: only react to the mic while listening or the user is
// speaking. During processing and agent playback the mic is ignored so we
// never transcribe the assistant's own voice.
function vadTick() {
  if (!callActive || !analyser) return;
  if (state !== "listening" && state !== "user_speaking") return;

  const vol = currentVolume();
  const now = Date.now();

  if (state === "listening") {
    if (vol > VAD_THRESHOLD) {
      loudCount += 1;
      if (loudCount >= SPEECH_START_CHECKS) beginUtterance(now);
    } else {
      loudCount = 0;
    }
  } else if (state === "user_speaking") {
    if (vol > VAD_THRESHOLD) {
      lastLoudTime = now;
    } else if (now - lastLoudTime >= SILENCE_MS) {
      endUtterance();
    }
  }
}

function beginUtterance(now) {
  chunks = [];
  try {
    recorder = recorderMime ? new MediaRecorder(micStream, { mimeType: recorderMime })
                            : new MediaRecorder(micStream);
  } catch (e) {
    recorder = new MediaRecorder(micStream);
  }
  recorder.addEventListener("dataavailable", (ev) => {
    if (ev.data && ev.data.size > 0) chunks.push(ev.data);
  });
  recorder.addEventListener("stop", () => {
    const durationMs = Date.now() - utteranceStart;
    const blob = new Blob(chunks, { type: recorderMime || "audio/webm" });
    if (durationMs < MIN_UTTERANCE_MS || blob.size === 0) {
      if (callActive) setState("listening");
      return;
    }
    sendUtterance(blob);
  });
  recorder.start();
  utteranceStart = now;
  lastLoudTime = now;
  loudCount = 0;
  setState("user_speaking");
}

function endUtterance() {
  setState("processing");
  if (recorder && recorder.state === "recording") {
    try { recorder.stop(); } catch (e) { /* ignore */ }
  }
}

function resumeListeningIfActive() {
  if (callActive) { loudCount = 0; setState("listening"); }
  else setState("ended");
}

// --- Send one utterance to the Mac backend ---
async function sendUtterance(blob) {
  setState("processing");
  const form = new FormData();
  form.append("audio", blob, "turn." + fileExtForMime(recorderMime));
  form.append("input_mode", "customer_continuous");
  form.append("stt_provider", sttSelect.value);
  form.append("tts_provider", ttsSelect.value);
  form.append("router_mode", routerSelect.value);
  form.append("prompt_mode", promptSelect.value);
  form.append("llm_model", modelInput.value || "llama3.2");
  form.append("session_id", getSessionId());

  let result;
  try {
    const resp = await fetch("/api/voice-turn", { method: "POST", body: form });
    result = await resp.json();
  } catch (e) {
    setState("error", "Could not reach the server.");
    setTimeout(resumeListeningIfActive, 2000);
    return;
  }

  if (!result || !result.success) {
    const msg = (result && result.error) ? result.error : "Something went wrong.";
    setState("error", msg);
    setTimeout(resumeListeningIfActive, 2500);
    return;
  }

  liveYou.textContent = "You: " + result.transcript;
  liveAgent.textContent = "Agent: " + result.response_to_customer;
  addToHistory(result);
  updateDebug(result);
  playAgentReply(result.audio_url);
}

function playAgentReply(url) {
  setState("agent_speaking");
  tapToPlayBtn.classList.add("hidden");
  agentAudio.src = url;
  agentAudio.onended = resumeListeningIfActive;

  const p = agentAudio.play();
  if (p && p.catch) {
    p.catch(() => {
      // Autoplay blocked: offer a manual play button, keep the call paused
      // here until the user taps it (then we resume listening after playback).
      tapToPlayBtn.classList.remove("hidden");
      setState("agent_speaking", "Tap to hear the agent");
    });
  }
}

tapToPlayBtn.addEventListener("click", () => {
  tapToPlayBtn.classList.add("hidden");
  const p = agentAudio.play();
  if (p && p.catch) p.catch(() => resumeListeningIfActive());
});

function addToHistory(result) {
  if (historyEl.dataset.hasTurns !== "true") {
    historyEl.innerHTML = "";
    historyEl.dataset.hasTurns = "true";
  }
  const turn = document.createElement("div");
  turn.className = "turn";
  const you = document.createElement("p");
  you.className = "you";
  you.textContent = "You: " + result.transcript;
  const agent = document.createElement("p");
  agent.className = "agent";
  agent.textContent = "Agent: " + result.response_to_customer;
  turn.appendChild(you);
  turn.appendChild(agent);
  historyEl.prepend(turn);
}

function updateDebug(result) {
  const out = result.llm_output || {};
  const finalRoute = out.final_route || out.top_level_route || "(clarify)";
  dbgRoute.textContent = finalRoute;
  dbgSource.textContent = result.routing_source
    + (result.routing_source === "pre_router" && result.rule_name ? " (" + result.rule_name + ")" : "");
  dbgLatency.textContent = (result.latency && result.latency.total != null)
    ? result.latency.total + "s" : "—";
}

// Which capture mode is selected: "blob" (stable) or "streaming" (experimental).
function getCaptureMode() {
  const checked = document.querySelector('input[name="capture-mode"]:checked');
  return checked ? checked.value : "blob";
}

startBtn.addEventListener("click", () => {
  if (getCaptureMode() === "streaming" && window.AICCStreaming) {
    window.AICCStreaming.start();   // experimental streaming STT
  } else {
    startCall();                    // stable continuous blob mode
  }
});

endBtn.addEventListener("click", () => {
  if (getCaptureMode() === "streaming" && window.AICCStreaming && window.AICCStreaming.isActive()) {
    window.AICCStreaming.end();
  } else {
    endCall("ended");
  }
});

// Small shared API so the streaming module can reuse the UI helpers and
// elements without duplicating them. Blob mode does not depend on this.
window.AICC = {
  setState,
  addToHistory,   // works with /api/text-turn results (they include transcript + response_to_customer)
  updateDebug,    // works with /api/text-turn results (they include llm_output, routing_source, latency)
  agentAudio,
  tapToPlayBtn,
  startBtn,
  endBtn,
  els: { liveInterim, liveYou, liveAgent },
  settings: { sttSelect, ttsSelect, routerSelect, promptSelect, modelInput, ttsOutputSelect },
  session: { start: startSession, get: getSessionId, end: endSession },
};
