import pyttsx3
import threading
import queue

class TextToSpeech:
    """
    Thread-safe, non-blocking TTS system
    FIXES:
    - No engine reuse crashes
    - No multi-thread overlap freeze
    - Queued speech (prevents audio spam crash)
    """

    def __init__(self):
        self.queue = queue.Queue()
        self.lock = threading.Lock()
        self.running = True

        self.worker_thread = threading.Thread(
            target=self._worker,
            daemon=True
        )
        self.worker_thread.start()

        print("✅ Text-to-Speech initialized (thread-safe queue mode)")

    def _create_engine(self):
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.setProperty("volume", 0.9)
        return engine

    def _worker(self):
        while self.running:
            text = self.queue.get()
            if text is None:
                break

            try:
                engine = self._create_engine()
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception as e:
                print(f"❌ TTS error: {e}")

    def speak(self, text: str):
        if not text or not text.strip():
            return

        print(f"🔊 Queued speech: {text}")
        self.queue.put(text.strip())

    def speak_async(self, text: str):
        # same behavior, kept for compatibility
        self.speak(text)

    def stop(self):
        self.running = False
        self.queue.put(None)


# ---------------- GLOBAL WRAPPER ---------------- #

tts_engine = None


def init_tts():
    global tts_engine
    if tts_engine is None:
        tts_engine = TextToSpeech()


def speak_text(text, use_async=True):
    global tts_engine

    if tts_engine is None:
        init_tts()

    if not text or not text.strip():
        return

    if use_async:
        tts_engine.speak_async(text)
    else:
        tts_engine.speak(text)