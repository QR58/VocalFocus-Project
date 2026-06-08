"""
controllers.py

Connects the Qt UI (via AppSignals and widgets) to the AudioPipeline:

    * LiveController:
        - Handles live purification.
        - Records mixture to recordings/mix/.
        - Runs separation + ECAPA matching.
        - Saves purified target speaker to recordings/purified/.

    * EnrollmentController:
        - Handles enrollment dialog recording.
        - Records enrollment WAV to recordings/enroll/.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QObject

from .pipeline import AudioPipeline
from audio import audio_config
from diarization import enrollment as enroll_db
from separation.separator import TargetSpeakerSeparator
from ui.signals import AppSignals

# ---------------------------------------------------- #
# LiveController
# ---------------------------------------------------- #

class LiveController(QObject):
    """
    Owns the live audio pipeline.

    Responsibilities
    ----------------
    * Listen to AppSignals (start_live/stop_live/focus_changed).
    * Start/stop AudioPipeline for the selected input device.
    * Record mixture to recordings/mix/mix_*.wav.
    * After stop, run SepFormer + ECAPA to keep only the target speaker.
    * Save purified audio to recordings/purified/purified_<name>_mix_*.wav.
    """

    def __init__(
        self,
        *,
        signals: AppSignals,
        level_meter,
        input_combo,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._signals = signals
        self._level_meter = level_meter
        self._input_combo = input_combo

        self._pipeline = AudioPipeline(parent=self)
        self._pipeline.level_changed.connect(self._level_meter.set_level)
        self._pipeline.error.connect(self._on_error)

        # Recorded mixture path
        self._current_mix_path: Optional[Path] = None

        # Target profile info
        self._target_profile_name: Optional[str] = None
        self._target_enroll_path: Optional[Path] = None

        # Heavy models (SepFormer + ECAPA) loaded once
        self._separator = TargetSpeakerSeparator()

        # Wire up signals
        self._signals.start_live.connect(self._on_start_live)
        self._signals.stop_live.connect(self._on_stop_live)
        self._signals.focus_changed.connect(self._on_focus_changed)

    # ------------------------------------------------------------------ #
    
    def _on_focus_changed(self, name: str) -> None:
        """
        Called when the UI tells us that the focused / target speaker
        has changed (via AppSignals.focus_changed).
        """
        self._target_profile_name = name
        profile = enroll_db.get_profile_by_name(name)
        if profile is None:
            self._target_enroll_path = None
            self._signals.log_message.emit(
                f"[Live] Focus set to '{name}', but no enrolled profile was found."
            )
            return

        self._target_enroll_path = Path(profile.enroll_wav)
        self._signals.log_message.emit(
            f"[Live] Target speaker set to '{name}' "
            f"(enrollment audio: {self._target_enroll_path.name})."
        )

    def _on_start_live(self) -> None:
        """Start live capture, recording mixture to recordings/mix/."""
        device_name = self._input_combo.currentText()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        mix_path = audio_config.LIVE_MIX_DIR / f"mix_{ts}.wav"
        self._current_mix_path = mix_path

        self._signals.log_message.emit(
            f"[Live] Starting capture on device '{device_name}', "
            f"recording mixture to {mix_path}"
        )

        self._pipeline.start(device_name=device_name, record_path=str(mix_path))

    def _on_stop_live(self) -> None:
        """Stop live capture and run target-speaker purification (if possible)."""
        self._pipeline.stop()

        mix_path = self._current_mix_path
        enroll_path = self._target_enroll_path

        # --- Check mixture file ------------------------------------------------
        if mix_path is None:
            self._signals.log_message.emit("[Live] No mixture file recorded; nothing to purify.")
            return

        if not mix_path.exists():
            self._signals.log_message.emit(f"[Live] Mixture file {mix_path} does not exist.")
            return

        if mix_path.stat().st_size == 0:
            self._signals.log_message.emit(
                f"[Live] Mixture file {mix_path} is empty (0 bytes); skipping purification."
            )
            return

        # --- Ensure we have an enrollment WAV ----------------------------------
        if enroll_path is None:
            profiles = enroll_db.load_profiles()
            if profiles:
                fallback = profiles[0]
                enroll_path = Path(fallback.enroll_wav)
                self._target_profile_name = fallback.name
                self._target_enroll_path = enroll_path
                self._signals.log_message.emit(
                    f"[Live] No focus set; falling back to first profile '{fallback.name}'."
                )
            else:
                self._signals.log_message.emit(
                    "[Live] No enrolled target available; skipping purification."
                )
                return

        if not enroll_path.exists():
            self._signals.log_message.emit(
                f"[Live] Enrollment file {enroll_path} does not exist; skipping purification."
            )
            return

        if enroll_path.stat().st_size == 0:
            self._signals.log_message.emit(
                f"[Live] Enrollment file {enroll_path} is empty (0 bytes); skipping purification."
            )
            return

        # --- Decide output path for purified audio -----------------------------
        base_name = mix_path.stem  # e.g. "mix_20251208_020000"
        profile_tag = self._target_profile_name or "target"
        out_name = f"purified_{profile_tag}_{base_name}.wav"

        # Prefer PURIFIED_DIR, but fall back to the mix folder if not defined
        try:
            out_dir = audio_config.PURIFIED_DIR
        except AttributeError:
            out_dir = mix_path.parent
        out_path = Path(out_dir) / out_name

        self._signals.log_message.emit(
            f"[Live] Starting purification:\n"
            f"       mix={mix_path} (size={mix_path.stat().st_size} bytes)\n"
            f"       enroll={enroll_path} (size={enroll_path.stat().st_size} bytes)\n"
            f"       output={out_path}"
        )

        # --- Run SpeechBrain models --------------------------------------------
        try:
            self._separator.separate_to_target(
                mixture_path=str(mix_path),
                enroll_wav_path=str(enroll_path),
                output_path=str(out_path),
            )
            self._signals.log_message.emit(
                f"[Live] Saved purified target-speaker WAV to: {out_path}"
            )
        except Exception as exc:
            # This will show up in your Logs page and the console.
            self._signals.log_message.emit(f"[Live] Purification failed: {exc!r}")

    def _on_error(self, msg: str) -> None:
        # surface pipeline errors into the shared log
        self._signals.log_message.emit(f"[AudioPipeline] {msg}")

# ------------------------------------------------ #
# EnrollmentController
# ------------------------------------------------ #


class EnrollmentController(QObject):
    """
    Controller used by the EnrollDialog.

    It owns a dedicated AudioPipeline for enrollment:
        * Shows live level in the dialog.
        * Records enrollment WAV to recordings/enroll/enroll_*.wav.
    """

    def __init__(self, *, level_meter, device_combo, parent=None) -> None:
        super().__init__(parent)
        self._meter = level_meter
        self._device_combo = device_combo

        self._pipeline = AudioPipeline(parent=self)
        self._pipeline.level_changed.connect(self._meter.set_level)

        self._recording: bool = False
        self._current_enroll_path: Optional[Path] = None

    # ------------------------------------------------------------------ #

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def last_recording_path(self) -> Optional[Path]:
        """
        Path to the most recent enrollment recording, or None
        if no enrollment has been recorded yet.
        """
        return self._current_enroll_path

    def toggle_recording(self) -> None:
        """
        Start/stop enrollment recording.

        When starting:
            * Create a new enroll_*.wav under recordings/enroll/.
            * Start AudioPipeline with that path as record_path.

        When stopping:
            * Stop the pipeline (which closes the WAV file).
        """
        if not self._recording:
            device_name = self._device_combo.currentText()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = audio_config.ENROLL_RECORDINGS_DIR / f"enroll_{ts}.wav"
            self._current_enroll_path = path

            self._pipeline.start(device_name=device_name, record_path=str(path))
            self._recording = True
        else:
            self._pipeline.stop()
            self._recording = False

    def stop(self) -> None:
        """Force-stop recording and reset meter."""
        if self._recording:
            self._pipeline.stop()
        self._recording = False
        self._meter.set_level(0.0)
