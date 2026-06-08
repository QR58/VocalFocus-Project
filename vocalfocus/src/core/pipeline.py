"""
pipeline.py

High-level audio pipeline built on top of `audio.capture.AudioCaptureWorker`.

Responsibilities:
    * Start/stop microphone capture for a given device.
    * Expose a UI-friendly `level_changed` signal.
    * Convert raw bytes to NumPy arrays.
    * Run light-weight preprocessing (pre-emphasis + normalize).
    * Optionally write raw chunks into a .wav file on disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import wave

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from audio import audio_config, audio_utils, preprocessor
from audio.capture import AudioCaptureWorker


class AudioPipeline(QObject):
    """
    Small, reusable audio pipeline.

    Signals
    -------
    level_changed(float)
        For VU / waveform meters (0–1).
    raw_chunk(bytes)
        Raw 16-bit PCM bytes from the microphone.
    processed_chunk(object)
        Preprocessed NumPy array (mono float32).
    error(str)
        Human-readable error message.
    """

    level_changed = pyqtSignal(float)
    raw_chunk = pyqtSignal(bytes)
    processed_chunk = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._capture: Optional[AudioCaptureWorker] = None

        # WAV recording
        self._wav_file: Optional[wave.Wave_write] = None
        self._record_path: Optional[Path] = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #


    def start(self,device_name: Optional[str] = None,*,record_path: Optional[str] = None) -> None:
        """
        Start capturing from the given device.

        Parameters
        ----------
        device_name : str | None
            Human-readable device name from QMediaDevices / combo box.
        record_path : str | None
            If provided, all raw audio chunks get written to this .wav file.
        """
         # If already running, stop first (also closes WAV file)
        if self._capture is not None:
            self.stop()

        # Prepare WAV writer if we want to record
        self._open_wav_writer(record_path)

        # Start low-level capture worker
        self._capture = AudioCaptureWorker(device_name=device_name, parent=self)
        self._capture.level_changed.connect(self.level_changed)
        self._capture.chunk_captured.connect(self._on_chunk_captured)
        self._capture.start()

    def stop(self) -> None:
        """Stop capturing, close the WAV file, reset the meter."""
        if self._capture is not None:
            self._capture.stop()
            self._capture = None

        self._close_wav_writer()

        # Reset level meter
        self.level_changed.emit(0.0)
    # ------------------------------------------------------------------ #
    # WAV helpers
    # ------------------------------------------------------------------ #

    def _open_wav_writer(self, record_path: Optional[str]) -> None:
        """Open a .wav file if `record_path` is given; otherwise disable recording."""
        if record_path is None:
            self._wav_file = None
            self._record_path = None
            return

        path = Path(record_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        wf = wave.open(str(path), "wb")
        wf.setnchannels(audio_config.CHANNELS)
        wf.setsampwidth(audio_config.SAMPLE_WIDTH)
        wf.setframerate(audio_config.SAMPLE_RATE)

        self._wav_file = wf
        self._record_path = path

    def _close_wav_writer(self) -> None:
        """Close the WAV file, if open."""
        if self._wav_file is not None:
            try:
                self._wav_file.close()
            except Exception:
                pass
        self._wav_file = None
        self._record_path = None

    # ------------------------------------------------------------------ #
    # Internal processing
    # ------------------------------------------------------------------ #

    def _on_chunk_captured(self, data: bytes) -> None:
        """
        Handle raw PCM data from AudioCaptureWorker.

        * Emits raw bytes.
        * Writes to WAV if enabled.
        * Runs basic preprocessing and emits float32 NumPy array.
        """
        # 1) forward raw bytes
        self.raw_chunk.emit(data)

        # 2) write to WAV if requested
        if self._wav_file is not None:
            try:
                self._wav_file.writeframes(data)
            except Exception as exc:
                self.error.emit(f"WAV write error: {exc}")

        # 3) preprocessing path: bytes -> float32 mono -> pre-emphasis -> normalize
        try:
            samples: np.ndarray = audio_utils.convert_to_array(data)

            # Non-streaming pre-emphasis, simple & robust
            samples = preprocessor.pre_emphasis(samples, coef=0.97)
            samples = audio_utils.normalize_audio(samples)

            self.processed_chunk.emit(samples)

        except Exception as exc:
            # Keep pipeline alive; just report error
            self.error.emit(str(exc))
