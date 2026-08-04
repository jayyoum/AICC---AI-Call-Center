// AICC Banking Assistant - EXPERIMENTAL streaming STT mode.
//
// Streams 16kHz mono Int16 PCM microphone chunks to the Flask backend over
// Socket.IO. The backend runs Google streaming recognition and emits interim /
// final transcripts. On a final transcript we call /api/text-turn (which skips
// STT, since it already happened) to route + synthesize the agent reply, then
// resume a fresh listening stream.
//
// This is a prototype. It reuses UI helpers exposed by customer.js on
// window.AICC, and defines window.AICCStreaming used by the Start/End buttons.
// The stable "blob" mode in customer.js is completely independent of this file.

(function () {
  const AICC = window.AICC || {};
  const setState = AICC.setState || function () {};
  const agentAudio = AICC.agentAudio;
  const tapToPlayBtn = AICC.tapToPlayBtn;
  const startBtn = AICC.startBtn;
  const endBtn = AICC.endBtn;
  const els = AICC.els || {};
  const settings = AICC.settings || {};
  const session = AICC.session || { start: () => null, get: () => null, end: () => {} };

  // One Google streaming session per utterance is used: we stop the stream on a
  // final transcript, process it, then start a new stream after the agent reply.
  let active = false;        // true between Start Call and End Call in streaming mode
  let socket = null;
  let micStream = null;
  let audioContext = null;
  let processor = null;
  let sourceNode = null;
  let zeroGain = null;
  let sttReady = false;      // server confirmed the stream is open
  let sendingAudio = false;  // gate: only stream mic audio while true
  let processingTurn = false; // true from final transcript until agent reply done
  let timings = {};

  // Streaming TTS state (experimental). Chunks are collected then played as one
  // WAV Blob ("streamed transport, buffered playback" - Strategy B).
  const DEFAULT_TTS_VOICE = "en-US-Chirp3-HD-Charon";
  let ttsChunks = [];
  let ttsSampleRate = 24000;
  let lastFinalTranscript = "";
  let turnResolved = false; // guards against double handling (done + error)
  let ttsTimings = {};

  function getTtsMode() {
    return settings.ttsOutputSelect ? settings.ttsOutputSelect.value : "file_tts";
  }

  // --- Audio conversion helpers (done in the browser) -----------------------

  // Average-decimation downsample from the mic's native rate to 16kHz.
  function downsampleTo16k(float32, inputRate) {
    if (inputRate === 16000) return float32;
    const ratio = inputRate / 16000;
    const outLength = Math.floor(float32.length / ratio);
    const out = new Float32Array(outLength);
    let iOut = 0;
    let iIn = 0;
    while (iOut < outLength) {
      const nextIn = Math.floor((iOut + 1) * ratio);
      let sum = 0;
      let count = 0;
      for (let i = iIn; i < nextIn && i < float32.length; i++) { sum += float32[i]; count++; }
      out[iOut] = count > 0 ? sum / count : 0;
      iOut++;
      iIn = nextIn;
    }
    return out;
  }

  // Float32 [-1,1] samples -> signed 16-bit little-endian PCM (ArrayBuffer).
  function floatTo16BitPCM(float32) {
    const buffer = new ArrayBuffer(float32.length * 2);
    const view = new DataView(buffer);
    for (let i = 0; i < float32.length; i++) {
      let s = Math.max(-1, Math.min(1, float32[i]));
      s = s < 0 ? s * 0x8000 : s * 0x7fff;
      view.setInt16(i * 2, s, true);
    }
    return buffer;
  }

  // --- Streaming TTS helpers (base64 PCM chunks -> playable WAV) -------------

  function base64ToUint8(b64) {
    const binary = atob(b64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return bytes;
  }

  function concatUint8(list) {
    let total = 0;
    list.forEach((a) => { total += a.length; });
    const out = new Uint8Array(total);
    let offset = 0;
    list.forEach((a) => { out.set(a, offset); offset += a.length; });
    return out;
  }

  // Wrap raw mono 16-bit PCM into a WAV Blob so any browser can play it.
  function buildWavBlob(pcmBytes, sampleRate) {
    const dataLen = pcmBytes.length;
    const buffer = new ArrayBuffer(44 + dataLen);
    const view = new DataView(buffer);
    const writeStr = (offset, str) => {
      for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
    };
    writeStr(0, "RIFF"); view.setUint32(4, 36 + dataLen, true); writeStr(8, "WAVE");
    writeStr(12, "fmt "); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
    view.setUint16(22, 1, true); view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
    writeStr(36, "data"); view.setUint32(40, dataLen, true);
    new Uint8Array(buffer, 44).set(pcmBytes);
    return new Blob([buffer], { type: "audio/wav" });
  }

  // --- Start / end the streaming call ---------------------------------------

  async function start() {
    if (typeof io === "undefined") {
      setState("error", "Streaming client failed to load. Use Continuous (stable) mode.");
      return;
    }
    startBtn.disabled = true;
    tapToPlayBtn.classList.add("hidden");
    session.start(); // fresh session id for this call
    setState("connecting");

    // Unlock audio playback within the tap gesture (mobile browsers).
    try {
      agentAudio.playsInline = true;
      agentAudio.setAttribute("playsinline", "");
    } catch (e) { /* ignore */ }

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
    sourceNode = audioContext.createMediaStreamSource(micStream);

    // ScriptProcessorNode is deprecated but fine for a prototype; it fires
    // onaudioprocess every bufferSize/sampleRate seconds (~85ms at 48kHz).
    processor = audioContext.createScriptProcessor(4096, 1, 1);
    zeroGain = audioContext.createGain();
    zeroGain.gain.value = 0; // keep the node running without echoing mic to speakers
    sourceNode.connect(processor);
    processor.connect(zeroGain);
    zeroGain.connect(audioContext.destination);

    processor.onaudioprocess = (event) => {
      if (!active || !sendingAudio || processingTurn) return;
      const input = event.inputBuffer.getChannelData(0);
      const down = downsampleTo16k(input, audioContext.sampleRate);
      const pcm = floatTo16BitPCM(down);
      if (socket && socket.connected) socket.emit("audio_chunk", pcm);
    };

    // Connect Socket.IO to the same origin that served the page.
    socket = io();
    socket.on("connect", startStream);
    socket.on("stt_started", () => {
      sttReady = true;
      sendingAudio = true;
      timings.streamStart = Date.now();
      setState("streaming_listening");
    });
    socket.on("stt_interim", (data) => {
      if (!active) return;
      if (!timings.firstInterim) timings.firstInterim = Date.now();
      if (els.liveInterim) els.liveInterim.textContent = "… " + (data.transcript || "");
      setState("receiving_transcript");
    });
    socket.on("stt_final", (data) => onFinalTranscript(data.transcript || ""));
    socket.on("stt_error", (data) => {
      setState("error", (data && data.error) ? data.error : "Streaming STT error.");
      // Give up on this stream; try a fresh one shortly if still on the call.
      sendingAudio = false;
      sttReady = false;
      if (active) setTimeout(() => { if (active && !processingTurn) startStream(); }, 2500);
    });
    socket.on("stt_stopped", () => { sttReady = false; });
    socket.on("connect_error", () => {
      setState("error", "Could not connect to the streaming server.");
      startBtn.disabled = false;
    });

    // Streaming TTS events (only used when tts_output_mode = streaming_tts).
    socket.on("tts_stream_started", (d) => {
      ttsSampleRate = (d && d.sample_rate_hertz) || 24000;
    });
    socket.on("tts_audio_chunk", (d) => {
      if (!d || !d.chunk_base64) return;
      if (!ttsTimings.firstChunk) ttsTimings.firstChunk = Date.now();
      ttsChunks.push(base64ToUint8(d.chunk_base64));
    });
    socket.on("tts_stream_done", (d) => playStreamedTts(d));
    socket.on("tts_stream_error", (d) => {
      if (turnResolved) return;
      setState("error", (d && d.error) ? d.error : "Streaming TTS failed.");
      fallbackFileTurn(lastFinalTranscript); // safe fallback to file-based TTS
    });

    active = true;
    endBtn.disabled = false;
    if (socket.connected) startStream();
  }

  function startStream() {
    if (!active || !socket) return;
    if (els.liveInterim) els.liveInterim.textContent = "";
    sttReady = false;
    sendingAudio = false;
    socket.emit("stt_start", {
      language_code: "en-US",
      sample_rate_hertz: 16000,
      input_mode: "streaming_customer",
      session_id: session.get(),
    });
  }

  function onFinalTranscript(text) {
    if (!active || processingTurn) return;
    if (!text.trim()) return; // ignore empty finals

    processingTurn = true;
    sendingAudio = false;
    sttReady = false;
    timings.finalTranscript = Date.now();
    socket.emit("stt_stop"); // close this utterance's Google stream

    if (els.liveInterim) els.liveInterim.textContent = "";
    if (els.liveYou) els.liveYou.textContent = "You: " + text;
    setState("processing_final_transcript");

    lastFinalTranscript = text;
    if (getTtsMode() === "streaming_tts") {
      startStreamingTtsTurn(text);   // experimental: /api/route-text + streamed audio
    } else {
      textTurn(text);                // stable: /api/text-turn (routing + file TTS)
    }
  }

  // Route + synthesize the final transcript WITHOUT re-running STT.
  async function textTurn(transcript) {
    timings.textTurnStart = Date.now();
    let result;
    try {
      const resp = await fetch("/api/text-turn", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          transcript: transcript,
          session_id: session.get(),
          input_mode: "streaming_customer_text",
          tts_provider: settings.ttsSelect ? settings.ttsSelect.value : "local",
          router_mode: settings.routerSelect ? settings.routerSelect.value : "pre_router",
          prompt_mode: settings.promptSelect ? settings.promptSelect.value : "compact_v2",
          llm_model: settings.modelInput ? (settings.modelInput.value || "llama3.2") : "llama3.2",
        }),
      });
      result = await resp.json();
    } catch (e) {
      setState("error", "Could not reach the server.");
      afterTurn();
      return;
    }

    if (!result || !result.success) {
      setState("error", (result && result.error) ? result.error : "Something went wrong.");
      afterTurn();
      return;
    }

    if (els.liveAgent) els.liveAgent.textContent = "Agent: " + result.response_to_customer;
    if (AICC.addToHistory) AICC.addToHistory(result);
    if (AICC.updateDebug) AICC.updateDebug(result);
    playReply(result.audio_url);
  }

  function playReply(url) {
    setState("agent_speaking");
    tapToPlayBtn.classList.add("hidden");
    agentAudio.src = url;
    agentAudio.onended = afterTurn; // resume listening after the reply
    const p = agentAudio.play();
    if (p && p.catch) {
      p.catch(() => {
        // Autoplay blocked: show the manual play button (customer.js's handler
        // plays the element; our onended still runs to resume listening).
        tapToPlayBtn.classList.remove("hidden");
        setState("agent_speaking", "Tap to hear the agent");
      });
    }
  }

  // --- Experimental streaming TTS turn --------------------------------------
  // Route the final transcript (no TTS) via /api/route-text, then ask the
  // backend to stream Google TTS audio chunks and play them buffered.
  async function startStreamingTtsTurn(transcript) {
    turnResolved = false;
    ttsChunks = [];
    ttsTimings = { routeTextStart: Date.now() };

    let result;
    try {
      const resp = await fetch("/api/route-text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          transcript: transcript,
          session_id: session.get(),
          input_mode: "streaming_customer_route_only",
          router_mode: settings.routerSelect ? settings.routerSelect.value : "pre_router",
          prompt_mode: settings.promptSelect ? settings.promptSelect.value : "compact_v2",
          llm_model: settings.modelInput ? (settings.modelInput.value || "llama3.2") : "llama3.2",
        }),
      });
      result = await resp.json();
    } catch (e) {
      setState("error", "Could not reach the server.");
      fallbackFileTurn(transcript);
      return;
    }

    if (!result || !result.success) {
      setState("error", (result && result.error) ? result.error : "Routing failed.");
      fallbackFileTurn(transcript);
      return;
    }

    if (els.liveAgent) els.liveAgent.textContent = "Agent: " + result.response_to_customer;
    if (AICC.addToHistory) AICC.addToHistory(result);
    if (AICC.updateDebug) AICC.updateDebug(result);

    // Kick off the streamed TTS for the routed reply.
    ttsTimings.ttsStreamStart = Date.now();
    setState("agent_speaking", "Preparing audio…");
    socket.emit("tts_stream_start", {
      text: result.response_to_customer,
      language_code: "en-US",
      voice_name: DEFAULT_TTS_VOICE,
      input_mode: "streaming_customer_tts",
      session_id: session.get(),
    });
  }

  // Build one WAV Blob from the collected PCM chunks and play it.
  function playStreamedTts(doneInfo) {
    if (turnResolved) return;
    turnResolved = true;

    if (!ttsChunks.length) {
      fallbackFileTurn(lastFinalTranscript); // nothing streamed -> file TTS
      return;
    }

    const rate = (doneInfo && doneInfo.sample_rate_hertz) || ttsSampleRate || 24000;
    const wavBlob = buildWavBlob(concatUint8(ttsChunks), rate);
    const url = URL.createObjectURL(wavBlob);

    if (doneInfo) {
      console.log(
        "[streaming TTS] chunks=%s bytes=%s time_to_first=%ss total=%ss playback=streaming_buffered",
        doneInfo.chunk_count, doneInfo.total_audio_bytes,
        doneInfo.time_to_first_audio_chunk_seconds, doneInfo.latency_seconds
      );
    }

    setState("agent_speaking");
    tapToPlayBtn.classList.add("hidden");
    agentAudio.src = url;
    agentAudio.onended = () => { URL.revokeObjectURL(url); afterTurn(); };
    const p = agentAudio.play();
    if (p && p.catch) {
      p.catch(() => {
        tapToPlayBtn.classList.remove("hidden");
        setState("agent_speaking", "Tap to hear the agent");
      });
    }
  }

  // Fall back to the stable file-based path for one utterance. Callers handle
  // de-duplication (tts_stream_error checks turnResolved before calling).
  function fallbackFileTurn(transcript) {
    turnResolved = true;
    if (!transcript) { afterTurn(); return; }
    textTurn(transcript);
  }

  // Called once an utterance turn fully completes (or errors): resume listening.
  function afterTurn() {
    processingTurn = false;
    if (active) {
      startStream();
    } else {
      setState("ended");
    }
  }

  function end() {
    active = false;
    sendingAudio = false;
    sttReady = false;
    processingTurn = false;
    if (socket) {
      try { socket.emit("stt_stop"); } catch (e) {}
      try { socket.disconnect(); } catch (e) {}
      socket = null;
    }
    if (processor) { try { processor.disconnect(); } catch (e) {} processor = null; }
    if (zeroGain) { try { zeroGain.disconnect(); } catch (e) {} zeroGain = null; }
    if (sourceNode) { try { sourceNode.disconnect(); } catch (e) {} sourceNode = null; }
    if (micStream) { micStream.getTracks().forEach((t) => t.stop()); micStream = null; }
    if (audioContext) { try { audioContext.close(); } catch (e) {} audioContext = null; }
    try { agentAudio.pause(); } catch (e) {}

    startBtn.disabled = false;
    endBtn.disabled = true;
    if (els.liveInterim) els.liveInterim.textContent = "";
    if (els.liveYou) els.liveYou.textContent = "";
    if (els.liveAgent) els.liveAgent.textContent = "";
    session.end(); // next call gets a fresh session id
    setState("ended");
  }

  window.AICCStreaming = {
    start,
    end,
    isActive: () => active,
  };
})();
