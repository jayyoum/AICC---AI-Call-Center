"""
Experimental streaming Speech-to-Text using Google Cloud streaming recognition.

One GoogleStreamingSession represents a single live microphone stream (one per
connected browser/phone). Audio chunks (16kHz mono LINEAR16 PCM) are pushed in
from the network thread; a background worker thread feeds them to Google's
streaming_recognize() and calls back with interim and final transcripts.

Authentication uses Application Default Credentials (gcloud auth
application-default login) - no API keys or service-account files.

This is a prototype: it is intentionally small and readable rather than a
production streaming pipeline.
"""

import queue
import threading


class GoogleStreamingSession:
    """Manage one Google streaming-recognition session driven by a chunk queue."""

    def __init__(
        self,
        language_code: str = "en-US",
        sample_rate_hertz: int = 16000,
        on_interim=None,
        on_final=None,
        on_error=None,
    ):
        self.language_code = language_code
        self.sample_rate_hertz = int(sample_rate_hertz)
        # Callbacks are supplied by the caller (app.py wires them to Socket.IO).
        self._on_interim = on_interim or (lambda text: None)
        self._on_final = on_final or (lambda text: None)
        self._on_error = on_error or (lambda message: None)

        self._queue: "queue.Queue[bytes | None]" = queue.Queue()
        self._closed = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        """Start the background recognition worker. Returns False if unavailable."""
        try:
            from google.cloud import speech  # noqa: F401  (import check only)
        except ImportError:
            self._on_error("google-cloud-speech is not installed. Run: pip install -r requirements.txt")
            return False

        self._thread = threading.Thread(target=self._run, name="google-streaming-stt", daemon=True)
        self._thread.start()
        return True

    def add_chunk(self, data: bytes) -> None:
        """Queue one PCM audio chunk from the browser (ignored once closed)."""
        if not self._closed.is_set() and data:
            self._queue.put(data)

    def stop(self) -> None:
        """Signal end of stream and unblock the worker's request generator."""
        if not self._closed.is_set():
            self._closed.set()
            self._queue.put(None)  # sentinel wakes the generator so it can return

    # --- internals ---------------------------------------------------------

    def _request_generator(self):
        """Yield Google StreamingRecognizeRequests from the queued audio chunks."""
        from google.cloud import speech

        while not self._closed.is_set():
            chunk = self._queue.get()
            if chunk is None:  # sentinel from stop()
                return
            yield speech.StreamingRecognizeRequest(audio_content=chunk)

    def _run(self) -> None:
        try:
            from google.cloud import speech

            client = speech.SpeechClient()
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.sample_rate_hertz,
                language_code=self.language_code,
                enable_automatic_punctuation=True,
            )
            streaming_config = speech.StreamingRecognitionConfig(
                config=config,
                interim_results=True,
                single_utterance=False,
            )

            responses = client.streaming_recognize(streaming_config, self._request_generator())
            for response in responses:
                if self._closed.is_set():
                    break
                for result in response.results:
                    if not result.alternatives:
                        continue
                    transcript = result.alternatives[0].transcript
                    if result.is_final:
                        self._on_final(transcript)
                    else:
                        self._on_interim(transcript)

        except Exception as e:
            # Report a readable message; never surface a raw traceback to clients.
            if not self._closed.is_set():
                self._on_error(f"Streaming STT failed: {e}")
