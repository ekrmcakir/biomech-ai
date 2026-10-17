import os
import subprocess
import threading
from typing import Optional


class VoiceCoach:
    """
    Asynchronous non-blocking English voice coach using macOS native TTS engine.
    Guides the user through sequential 360° front, back, and side profile scans.
    """

    def __init__(self, enabled: bool = True, voice: str = "Samantha") -> None:
        self.enabled = enabled
        self.voice = voice
        self.current_process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def speak(self, text: str) -> None:
        """Speaks the provided text asynchronously without stuttering the video feed."""
        if not self.enabled:
            return

        def _run() -> None:
            with self._lock:
                try:
                    if self.current_process and self.current_process.poll() is None:
                        self.current_process.terminate()
                        try:
                            self.current_process.wait(timeout=0.3)
                        except subprocess.TimeoutExpired:
                            self.current_process.kill()

                    cmd = ["say", "-v", self.voice, text]
                    self.current_process = subprocess.Popen(cmd)
                except Exception:
                    try:
                        cmd = ["say", text]
                        self.current_process = subprocess.Popen(cmd)
                    except Exception:
                        pass

        threading.Thread(target=_run, daemon=True).start()

    def announce_scan_start(self) -> None:
        self.speak("Voice-guided body scan initiated. Please stand upright facing the camera.")

    def announce_front_captured(self) -> None:
        self.speak("Front scan complete! Now, please turn around to show your back.")

    def announce_back_captured(self) -> None:
        self.speak("Back scan complete! Now, please turn ninety degrees to the side.")

    def announce_all_completed(
        self,
        v_taper: Optional[float] = None,
        drop_side: Optional[str] = None,
        drop_cm: float = 0.0
    ) -> None:
        summary = "All scans completed successfully! Your three-dimensional posture and body composition report is ready."
        if v_taper and v_taper >= 1.40:
            summary += f" Your V-Taper ratio is {v_taper:.2f}, indicating an aesthetic athletic build."
        if drop_side and drop_side != "none" and drop_cm > 1.5:
            summary += f" Noticeable {drop_side} shoulder drop of approximately {drop_cm:.1f} centimeters detected."
        self.speak(summary)
