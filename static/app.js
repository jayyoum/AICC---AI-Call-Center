// AICC Voice Routing Prototype - browser-side recording and turn handling.
// Vanilla JS, no frameworks.
//
// Two input modes:
//   Manual:     click Start Recording / Stop Recording for each turn.
//   Continuous: click Start Conversation once; browser-side voice activity
//               detection (VAD) finds each utterance automatically.
//
// Both modes send one Blob per utterance to the same /api/voice-turn endpoint.

// ---------------------------------------------------------------------------
// VAD tuning constants (safe defaults, easy to adjust here or via the UI)
// ---------------------------------------------------------------------------
const VAD_CHECK_INTERVAL_MS = 50;    // how often we sample mic volume
const SPEECH_START_CHECKS = 3;       // consecutive loud checks (~150ms) to count as "speech started"
const MIN_UTTERANCE_MS = 400;        // ignore blips shorter than this (door slam, cough)
const ERROR_RESUME_DELAY_MS = 2500;  // in continuous mode, wait this long after an error before listening again
// Defaults for the two user-tunable values (also set in index.html):
//   VAD sensitivity (RMS threshold): 0.02   - lower = more sensitive
//   Silence duration: 900ms                 - how long a pause ends the utterance

// ---------------------------------------------------------------------------
// DOM references
// ---------------------------------------------------------------------------
const recordButton = document.getElementById("record-button");
const conversationButton = document.getElementById("conversation-button");
const statusIndicator = document.getElementById("status-indicator");
const stateBadge = document.getElementById("state-badge");
const transcriptPanel = document.getElementById("transcript-panel");
const responsePanel = document.getElementById("response-panel");
const responseAudio = document.getElementById("response-audio");
const historyPanel = document.getElementById("history-panel");

const tabManual = document.getElementById("tab-manual");
const tabContinuous = document.getElementById("tab-continuous");
const manualControls = document.getElementById("manual-controls");
const continuousControls = document.getElementById("continuous-controls");

const vadSensitivitySlider = document.getElementById("vad-sensitivity");
const vadSensitivityValue = document.getElementById("vad-sensitivity-value");
const silenceDurationInput = document.getElementById("silence-duration");

const routeTopLevel = document.getElementById("route-top-level");
const routeFinal = document.getElementById("route-final");
const routeAction = document.getElementById("route-action");
const routeConfidence = document.getElementById("route-confidence");

const latencyStt = document.getElementById("latency-stt");
const latencyLlm = document.getElementById("latency-llm");
const latencyTts = document.getElementById("latency-tts");
const latencyTotal = document.getElementById("latency-total");

const sttProviderSelect = document.getElementById("stt-provider");
const ttsProviderSelect = document.getElementById("tts-provider");
const llmModelInput = document.getElementById("llm-model");
const promptModeSelect = document.getElementById("prompt-mode");
const routerModeSelect = document.getElementById("router-mode");

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------
// Frontend state machine (used mainly by continuous mode, mirrored in the UI):
//   idle -> listening -> user_speaking -> processing -> speaking -> listening ...
//   any state can go to error, and error returns to listening (continuous) or idle (manual).
let currentState = "idle";

let inputMode = "manual"; // "manual" | "continuous"

// Manual mode
let manualRecorder = null;
let manualChunks = [];
let isManualRecording = false;

// Continuous mode
let conversationActive = false; // true between Start Conversation and Stop Conversation
let micStream = null;           // persistent mic stream while conversation is active
let audioContext = null;
let analyser = null;
let vadTimer = null;            // setInterval handle for the VAD loop
let utteranceRecorder = null;   // one MediaRecorder per detected utterance
let utteranceChunks = [];
let loudCheckCount = 0;         // consecutive loud samples while listening
let lastLoudTime = 0;           // last time we heard the user while they were speaking
let utteranceStartTime = 0;

// ---------------------------------------------------------------------------
// State + status display
// ---------------------------------------------------------------------------
function setState(newState) {
  currentState = newState;
  stateBadge.textContent = newState;
  stateBadge.className = "state-badge state-" + newState;

  const statusByState = {
    idle: ["Idle", "status-idle"],
    listening: ["Listening...", "status-listening"],
    user_speaking: ["Speaking detected...", "status-recording"],
    processing: ["Processing (STT -> LLM -> TTS)...", "status-processing"],
    speaking: ["Playing response...", "status-speaking"],
    error: ["Error", "status-error"],
  };
  const [text, cssClass] = statusByState[newState] || statusByState.idle;
  // Error text is set separately with the real message; don't overwrite it.
  if (newState !== "error") {
    setStatus(text, cssClass);
  }
}

function setStatus(text, cssClass) {
  statusIndicator.textContent = text;
  statusIndicator.className = "status " + cssClass;
}

function resetResultPanels() {
  transcriptPanel.textContent = "(no transcript yet)";
  responsePanel.textContent = "(no response yet)";
  routeTopLevel.textContent = "-";
  routeFinal.textContent = "-";
  routeAction.textContent = "-";
  routeConfidence.textContent = "-";
  latencyStt.textContent = "-";
  latencyLlm.textContent = "-";
  latencyTts.textContent = "-";
  latencyTotal.textContent = "-";
}

// ---------------------------------------------------------------------------
// Mode tabs
// ---------------------------------------------------------------------------
function switchMode(mode) {
  // Stop whatever the old mode was doing before switching.
  if (isManualRecording) stopManualRecording();
  if (conversationActive) stopConversation();

  inputMode = mode;
  tabManual.classList.toggle("active", mode === "manual");
  tabContinuous.classList.toggle("active", mode === "continuous");
  manualControls.classList.toggle("hidden", mode !== "manual");
  continuousControls.classList.toggle("hidden", mode !== "continuous");
  stateBadge.classList.toggle("hidden", mode !== "continuous");
  setState("idle");
}

tabManual.addEventListener("click", () => switchMode("manual"));
tabContinuous.addEventListener("click", () => switchMode("continuous"));

vadSensitivitySlider.addEventListener("input", () => {
  vadSensitivityValue.textContent = vadSensitivitySlider.value;
});

// ---------------------------------------------------------------------------
// Manual mode (unchanged flow: click to start, click to stop)
// ---------------------------------------------------------------------------
async function startManualRecording() {
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    setStatus("Microphone permission denied", "status-error");
    return;
  }

  manualChunks = [];
  manualRecorder = new MediaRecorder(stream);

  manualRecorder.addEventListener("dataavailable", (event) => {
    if (event.data && event.data.size > 0) {
      manualChunks.push(event.data);
    }
  });

  manualRecorder.addEventListener("stop", () => {
    // Stop the mic indicator/light once we're done recording.
    stream.getTracks().forEach((track) => track.stop());
    const audioBlob = new Blob(manualChunks, { type: "audio/webm" });
    sendTurn(audioBlob, "manual");
  });

  manualRecorder.start();
  isManualRecording = true;
  recordButton.textContent = "Stop Recording";
  recordButton.classList.add("recording");
  setStatus("Recording...", "status-recording");
}

function stopManualRecording() {
  if (manualRecorder && isManualRecording) {
    manualRecorder.stop();
    isManualRecording = false;
    recordButton.textContent = "Start Recording";
    recordButton.classList.remove("recording");
  }
}

recordButton.addEventListener("click", () => {
  if (isManualRecording) {
    setStatus("Stopping...", "status-processing");
    stopManualRecording();
  } else {
    startManualRecording();
  }
});

// ---------------------------------------------------------------------------
// Continuous mode: VAD loop + per-utterance recording
// ---------------------------------------------------------------------------
async function startConversation() {
  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    setStatus("Microphone permission denied", "status-error");
    return;
  }

  // Web Audio analyser gives us a cheap live volume estimate for VAD.
  audioContext = new (window.AudioContext || window.webkitAudioContext)();
  const source = audioContext.createMediaStreamSource(micStream);
  analyser = audioContext.createAnalyser();
  analyser.fftSize = 2048;
  source.connect(analyser);

  conversationActive = true;
  loudCheckCount = 0;
  conversationButton.textContent = "Stop Conversation";
  conversationButton.classList.add("recording");
  setState("listening");

  vadTimer = setInterval(vadCheck, VAD_CHECK_INTERVAL_MS);
}

function stopConversation() {
  conversationActive = false;

  if (vadTimer) {
    clearInterval(vadTimer);
    vadTimer = null;
  }
  // If an utterance was mid-recording, discard it rather than sending a cut-off turn.
  if (utteranceRecorder && utteranceRecorder.state === "recording") {
    utteranceRecorder.ondataavailable = null;
    utteranceRecorder.onstop = null;
    utteranceRecorder.stop();
  }
  utteranceRecorder = null;

  if (micStream) {
    micStream.getTracks().forEach((track) => track.stop());
    micStream = null;
  }
  if (audioContext) {
    audioContext.close();
    audioContext = null;
    analyser = null;
  }

  // Stop any response audio that's still playing.
  responseAudio.pause();

  conversationButton.textContent = "Start Conversation";
  conversationButton.classList.remove("recording");
  setState("idle");
}

conversationButton.addEventListener("click", () => {
  if (conversationActive) {
    stopConversation();
  } else {
    startConversation();
  }
});

// Returns the current mic volume as RMS in 0..1 (speech is typically 0.02-0.3).
function currentMicVolume() {
  const samples = new Uint8Array(analyser.fftSize);
  analyser.getByteTimeDomainData(samples);
  let sumOfSquares = 0;
  for (let i = 0; i < samples.length; i++) {
    const normalized = (samples[i] - 128) / 128;
    sumOfSquares += normalized * normalized;
  }
  return Math.sqrt(sumOfSquares / samples.length);
}

// Runs every VAD_CHECK_INTERVAL_MS while the conversation is active.
// Feedback-loop guard: mic input is only evaluated in the "listening" and
// "user_speaking" states. While the AI response is playing (state "speaking")
// or a request is in flight (state "processing"), VAD is ignored entirely,
// so the system never records or transcribes its own TTS output.
function vadCheck() {
  if (!conversationActive || !analyser) return;
  if (currentState !== "listening" && currentState !== "user_speaking") return;

  const volume = currentMicVolume();
  const threshold = parseFloat(vadSensitivitySlider.value);
  const silenceMs = parseInt(silenceDurationInput.value, 10) || 900;
  const now = Date.now();

  if (currentState === "listening") {
    if (volume > threshold) {
      loudCheckCount += 1;
      if (loudCheckCount >= SPEECH_START_CHECKS) {
        beginUtterance(now);
      }
    } else {
      loudCheckCount = 0;
    }
  } else if (currentState === "user_speaking") {
    if (volume > threshold) {
      lastLoudTime = now;
    } else if (now - lastLoudTime >= silenceMs) {
      endUtterance(now);
    }
  }
}

function beginUtterance(now) {
  utteranceChunks = [];
  // A fresh MediaRecorder per utterance keeps each turn a clean standalone Blob.
  utteranceRecorder = new MediaRecorder(micStream);

  utteranceRecorder.addEventListener("dataavailable", (event) => {
    if (event.data && event.data.size > 0) {
      utteranceChunks.push(event.data);
    }
  });

  utteranceRecorder.addEventListener("stop", () => {
    const durationMs = Date.now() - utteranceStartTime;
    const audioBlob = new Blob(utteranceChunks, { type: "audio/webm" });

    // Too short to be real speech (a cough, a bump): go straight back to listening.
    if (durationMs < MIN_UTTERANCE_MS || audioBlob.size === 0) {
      if (conversationActive) setState("listening");
      return;
    }
    sendTurn(audioBlob, "continuous");
  });

  utteranceRecorder.start();
  utteranceStartTime = now;
  lastLoudTime = now;
  loudCheckCount = 0;
  setState("user_speaking");
}

function endUtterance(now) {
  setState("processing");
  if (utteranceRecorder && utteranceRecorder.state === "recording") {
    utteranceRecorder.stop();
  }
}

// Called after a turn finishes (successfully or not) to keep the loop going.
function resumeListeningIfActive() {
  if (conversationActive) {
    loudCheckCount = 0;
    setState("listening");
  } else {
    setState("idle");
  }
}

// ---------------------------------------------------------------------------
// Shared turn handling (both modes)
// ---------------------------------------------------------------------------
async function sendTurn(audioBlob, mode) {
  setState("processing");
  resetResultPanels();
  recordButton.disabled = true;

  const formData = new FormData();
  formData.append("audio", audioBlob, "turn.webm");
  formData.append("stt_provider", sttProviderSelect.value);
  formData.append("tts_provider", ttsProviderSelect.value);
  formData.append("llm_model", llmModelInput.value || "llama3.2");
  formData.append("input_mode", mode);
  formData.append("prompt_mode", promptModeSelect.value || "compact_v2");
  formData.append("router_mode", routerModeSelect.value || "pre_router");

  let result;
  try {
    const response = await fetch("/api/voice-turn", {
      method: "POST",
      body: formData,
    });
    result = await response.json();
  } catch (err) {
    handleTurnError("Network error - could not reach the server", mode);
    return;
  }

  recordButton.disabled = false;

  if (!result.success) {
    handleTurnError(result.error, mode);
    return;
  }

  displayResult(result);
  addToHistory(result);

  // Play the response. VAD stays paused (state "speaking") until playback ends,
  // then continuous mode automatically resumes listening.
  setState("speaking");
  responseAudio.src = result.audio_url;
  responseAudio.onended = resumeListeningIfActive;
  responseAudio.play().catch(() => {
    // Autoplay can be blocked; don't stall the conversation loop waiting forever.
    resumeListeningIfActive();
  });
}

function handleTurnError(message, mode) {
  recordButton.disabled = false;
  setState("error");
  setStatus("Error: " + message, "status-error");

  // In continuous mode, show the error briefly and then keep the conversation going.
  if (mode === "continuous" && conversationActive) {
    setTimeout(() => {
      if (conversationActive && currentState === "error") {
        resumeListeningIfActive();
      }
    }, ERROR_RESUME_DELAY_MS);
  }
}

function displayResult(result) {
  transcriptPanel.textContent = result.transcript;
  responsePanel.textContent = result.response_to_customer;

  const llmOutput = result.llm_output || {};
  routeTopLevel.textContent = llmOutput.top_level_route || "-";
  routeFinal.textContent = llmOutput.final_route || "-";
  routeAction.textContent = llmOutput.action || "-";
  routeConfidence.textContent =
    llmOutput.confidence !== undefined && llmOutput.confidence !== null
      ? llmOutput.confidence
      : "-";

  latencyStt.textContent = result.latency.stt;
  latencyLlm.textContent = result.latency.llm;
  latencyTts.textContent = result.latency.tts;
  latencyTotal.textContent = result.latency.total;
}

function addToHistory(result) {
  // Clear the "No turns yet." placeholder on the first real turn.
  if (historyPanel.dataset.hasTurns !== "true") {
    historyPanel.innerHTML = "";
    historyPanel.dataset.hasTurns = "true";
  }

  const entry = document.createElement("div");
  entry.className = "history-entry";
  entry.innerHTML = `
    <p><strong>You said:</strong> ${escapeHtml(result.transcript)}</p>
    <p><strong>Route:</strong> ${escapeHtml(result.llm_output.top_level_route || "-")}
      &rarr; ${escapeHtml(result.llm_output.final_route || "-")}
      (${escapeHtml(result.llm_output.action || "-")})</p>
    <p><strong>Agent:</strong> ${escapeHtml(result.response_to_customer)}</p>
    <p class="history-meta">
      Mode: ${escapeHtml(result.input_mode || "manual")} |
      Routed by: ${escapeHtml(result.routing_source || "ollama")}${
        result.routing_source === "pre_router" && result.rule_name
          ? " (" + escapeHtml(result.rule_name) + ")"
          : " (" + escapeHtml(result.providers.prompt_mode || "compact") + ")"
      } |
      STT: ${result.providers.stt} | TTS: ${result.providers.tts} |
      Model: ${escapeHtml(result.providers.llm_model)} |
      Latency: STT ${result.latency.stt}s / route ${result.latency.route}s
      (LLM ${result.latency.llm}s) / TTS ${result.latency.tts}s / total ${result.latency.total}s
    </p>
  `;
  historyPanel.prepend(entry);
}

function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = value == null ? "" : String(value);
  return div.innerHTML;
}

// Dev dashboard only: show the full customer-UI URL for the current host,
// so it's easy to copy onto a phone on the same network.
const customerHostUrl = document.getElementById("customer-host-url");
if (customerHostUrl) {
  customerHostUrl.textContent =
    window.location.protocol + "//" + window.location.host + "/customer";
}
