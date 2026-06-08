"""
capture.py

Low-level microphone capture using PyAudio.

This module owns the direct interaction with the audio hardware.  GUI code
should never talk to PyAudio directly; it should use this worker (via
`pipeline.AudioPipeline` and the controller classes in `controllers.py`).
"""

from __future__ import annotations

import audioop
import logging
from typing import Optional

import pyaudio
from PyQt6.QtCore import QThread, pyqtSignal

from . import audio_config

logger = logging.getLogger(__name__)


class AudioCaptureWorker(QThread):
    """
    Background worker that pulls audio from an input device.

    Emits
    -----
    level_changed(float)
        Simple RMS-based meter level in [0, 1] for UI widgets.
    chunk_captured(bytes)
        Raw 16-bit PCM audio frames for further processing.
    """

    level_changed = pyqtSignal(float)
    chunk_captured = pyqtSignal(bytes)

    def __init__(self, device_name: Optional[str] = None, parent=None) -> None:
        super().__init__(parent)
        self.device_name = device_name
        self._running = False

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #

    def update_device(self, device_name: str) -> None:
        """Change the target device before starting the thread."""
        self.device_name = device_name

    def stop(self) -> None:
        """Ask the worker to stop and block until it finishes."""
        self._running = False
        self.wait()

    # ------------------------------------------------------------------ #
    # QThread implementation
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        pa = pyaudio.PyAudio()
        stream = None
        try:
            device_index = self._resolve_device_index(pa)

            stream = pa.open(
                format=pyaudio.paInt16,
                channels=audio_config.CHANNELS,
                rate=audio_config.SAMPLE_RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=audio_config.CHUNK_SIZE,
            )

            logger.info("AudioCaptureWorker started (device index=%s)", device_index)
            self._running = True

            while self._running:
                data = stream.read(audio_config.CHUNK_SIZE, exception_on_overflow=False)

                # raw bytes for downstream processing
                self.chunk_captured.emit(data)

                # light-weight RMS-based level for UI meters
                rms = audioop.rms(data, 2)  # 16-bit samples
                level = min(rms / 1500.0, 1.0)
                self.level_changed.emit(level)

        except Exception as exc:
            logger.exception("Error in AudioCaptureWorker: %s", exc)
        finally:
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
            pa.terminate()
            logger.info("AudioCaptureWorker stopped")

    # ------------------------------------------------------------------ #

    def _resolve_device_index(self, pa: "pyaudio.PyAudio") -> int:
        """
        Try to find a device index matching `self.device_name`.

        If no name is set or no match is found, the default input device
        is returned.
        """
        try:
            if not self.device_name:
                return pa.get_default_input_device_info()["index"]

            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if info.get("maxInputChannels", 0) <= 0:
                    continue
                name = info.get("name", "")
                if self.device_name in name:
                    logger.debug("Using input device %r (index=%s)", name, i)
                    return i

            logger.warning(
                "Requested device %r not found; falling back to default input",
                self.device_name,
            )
            return pa.get_default_input_device_info()["index"]
        except Exception as exc:  # defensive fallback
            logger.exception("Failed to resolve device index: %s", exc)
            return pa.get_default_input_device_info()["index"]
