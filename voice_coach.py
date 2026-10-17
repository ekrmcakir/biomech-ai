import os
import time
import subprocess
import threading


class VoiceCoach:
    """
    Asynchronous non-blocking English voice coach using macOS native TTS engine.
    Guides the user through sequential 360° front, back, and side profile scans.
    """

    def __init__(self, enabled=True, voice="Samantha"):
        self.enabled = enabled
        self.voice = voice
        self.current_process = None

    def speak(self, text):
        """Speaks the provided text asynchronously without stuttering the video feed."""
        if not self.enabled:
            return

        def _run():
            try:
                if self.current_process and self.current_process.poll() is None:
                    self.current_process.terminate()

                cmd = ["say", "-v", self.voice, text]
                self.current_process = subprocess.Popen(cmd)
                self.current_process.wait()
            except Exception:
                try:
                    cmd = ["say", text]
                    self.current_process = subprocess.Popen(cmd)
                    self.current_process.wait()
                except Exception:
                    pass

        threading.Thread(target=_run, daemon=True).start()

    def announce_scan_start(self):
        self.speak("Voice-guided body scan initiated. Please stand upright facing the camera.")

    def announce_front_captured(self):
        self.speak("Front scan complete! Now, please turn around to show your back.")

    def announce_back_captured(self):
        self.speak("Back scan complete! Now, please turn ninety degrees to the side.")

    def announce_all_completed(self, v_taper=None, drop_side=None, drop_cm=0.0):
        summary = "All scans completed successfully! Your three-dimensional posture and body composition report is ready."
        if v_taper and v_taper >= 1.40:
            summary += f" Your V-Taper ratio is {v_taper:.2f}, indicating an aesthetic athletic build."
        if drop_side and drop_side != "none" and drop_cm > 1.5:
            summary += f" Noticeable {drop_side} shoulder drop of approximately {drop_cm:.1f} centimeters detected."
        self.speak(summary)
