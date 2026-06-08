from __future__ import annotations

from pathlib import Path
from typing import Optional, List

import logging
import torch
import torchaudio
import torchaudio.functional as F
from speechbrain.inference.separation import SepformerSeparation
from speechbrain.inference.speaker import SpeakerRecognition


logger = logging.getLogger(__name__)


class TargetSpeakerSeparator:
    """
    Combines SpeechBrain SepFormer (source separation) and ECAPA-TDNN
    (speaker verification) to keep only the target speaker.

    Pipeline (high level):
        1. Separate mixture into N sources with SepFormer.
        2. Save each separated source to a temporary WAV file.
        3. For each candidate, compute ECAPA score against enrollment WAV.
        4. Select the source with the best score and save it as `output_path`.
    """

    # SepFormer WHAM models expect 8 kHz single-channel audio.
    # See model card: "The system expects input recordings sampled at 8kHz"
    # https://huggingface.co/speechbrain/sepformer-wham
    SEPFORMER_SAMPLE_RATE: int = 8000

    def __init__(self, device: Optional[str] = None) -> None:
        # Decide device once and reuse.
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("[Separator] Initializing TargetSpeakerSeparator on device=%s", self.device)

        # SepFormer multi-speaker separation model.
        # NOTE: We deliberately DO NOT pass `savedir` on Windows to avoid symlink issues.
        # The model will be stored & reused from the default Hugging Face cache
        # (e.g. C:\\Users\\<you>\\.cache\\huggingface\\hub).
        self._sep_model = SepformerSeparation.from_hparams(
            source="speechbrain/sepformer-wsj02mix",
            run_opts={"device": self.device},
        )
        logger.info("[Separator] Loaded SepFormer model: speechbrain/sepformer-wsj02mix")

        # ECAPA-TDNN speaker verification model.
        self._spk_model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": self.device},
        )
        logger.info("[Separator] Loaded SpeakerRecognition model: speechbrain/spkrec-ecapa-voxceleb")

    # ------------------------------------------------------------------ #
    # Main public API
    # ------------------------------------------------------------------ #

    def separate_to_target(
        self,
        *,
        mixture_path: str,
        enroll_wav_path: str,
        output_path: str,
    ) -> None:
        """
        Separate the mixture and keep the source that best matches
        the enrolled speaker (according to ECAPA scores).

        Parameters
        ----------
        mixture_path : str
            Path to mixture WAV (recordings/mix/mix_*.wav).
        enroll_wav_path : str
            Path to enrollment WAV (recordings/enroll/enroll_*.wav).
        output_path : str
            Final purified WAV will be written here.
        """
        # Resolve to absolute paths early (avoids weird relative-path behaviour
        # such as "D:Download\\..."). Then convert to POSIX strings
        # ("D:/Download/...") before passing into SpeechBrain / libsndfile.
        mix_path = Path(mixture_path).expanduser().resolve()
        enroll_path = Path(enroll_wav_path).expanduser().resolve()
        out_path = Path(output_path).expanduser().resolve()

        logger.info(
            "[Separator] separate_to_target called with:\n"
            "   mix=%s\n"
            "   enroll=%s\n"
            "   output=%s",
            mix_path,
            enroll_path,
            out_path,
        )

        # Basic existence checks
        if not mix_path.exists():
            logger.error("[Separator] Mixture file not found: %s", mix_path)
            raise FileNotFoundError(f"Mixture file not found: {mix_path}")

        if not enroll_path.exists():
            logger.error("[Separator] Enrollment file not found: %s", enroll_path)
            raise FileNotFoundError(f"Enrollment file not found: {enroll_path}")

        mix_size = mix_path.stat().st_size
        enroll_size = enroll_path.stat().st_size
        logger.info(
            "[Separator] File sizes: mix=%d bytes, enroll=%d bytes",
            mix_size,
            enroll_size,
        )

        if mix_size == 0:
            raise RuntimeError(f"Mixture file {mix_path} is empty (0 bytes).")
        if enroll_size == 0:
            raise RuntimeError(f"Enrollment file {enroll_path} is empty (0 bytes).")

        # Prepare output directory
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to POSIX-style paths for all external libraries
        mix_str = mix_path.as_posix()
        enroll_str = enroll_path.as_posix()
        out_str = out_path.as_posix()

        logger.debug(
            "[Separator] Using POSIX paths:\n"
            "   mix=%s\n"
            "   enroll=%s\n"
            "   output=%s",
            mix_str,
            enroll_str,
            out_str,
        )

        # 1) Separate mixture into sources
        try:
            logger.info("[Separator] Running SepFormer separation on mixture.")
            est_sources = self._sep_model.separate_file(path=mix_str)
        except Exception as exc:
            # This will surface details in the Logs window.
            logger.exception(
                "[Separator] SepFormer separation failed for mix=%s: %r",
                mix_str,
                exc,
            )
            raise

        if est_sources.ndim != 3 or est_sources.shape[0] != 1:
            logger.error(
                "[Separator] Unexpected separated tensor shape: %s",
                tuple(est_sources.shape),
            )
            raise RuntimeError(f"Unexpected separated tensor shape: {est_sources.shape}")

        _, time_steps, num_sources = est_sources.shape
        logger.info(
            "[Separator] Separation output shape: batch=1, time=%d, num_sources=%d",
            time_steps,
            num_sources,
        )

        if num_sources < 1:
            raise RuntimeError("Separation model returned zero sources.")

        # 2) Evaluate each candidate using ECAPA verification
        tmp_paths: List[Path] = []
        best_score: Optional[float] = None
        best_tmp_path: Optional[Path] = None

        ECAPA_SR = 16000  # ECAPA expects 16 kHz

        for idx in range(num_sources):
            # Candidate waveform at 8 kHz: shape [1, time]
            src_8k = est_sources[0, :, idx].unsqueeze(0).cpu()
            logger.debug(
                "[Separator] Evaluating candidate source %d with waveform shape=%s",
                idx,
                tuple(src_8k.shape),
            )

            # --- (1) Resample candidate from 8 kHz -> 16 kHz for ECAPA ---
            try:
                src_16k = F.resample(
                    src_8k,
                    self.SEPFORMER_SAMPLE_RATE,
                    ECAPA_SR,
                )
            except Exception as exc:
                logger.exception(
                    "[Separator] Failed to resample candidate %d: %r",
                    idx,
                    exc,
                )
                continue

            tmp_path = out_path.with_name(out_path.stem + f"_cand{idx}.wav")
            tmp_str = tmp_path.as_posix()

            # Save candidate at 16 kHz so ECAPA sees the correct sampling rate
            try:
                torchaudio.save(tmp_str, src_16k, ECAPA_SR)
            except Exception as exc:
                logger.exception(
                    "[Separator] Failed to save candidate WAV %s: %r", tmp_str, exc
                )
                continue

            tmp_paths.append(tmp_path)

            # --- (2) ECAPA speaker verification at 16 kHz ---
            try:
                score, prediction = self._spk_model.verify_files(
                    enroll_str,
                    tmp_str,
                )
            except Exception as exc:
                logger.exception(
                    "[Separator] ECAPA verification failed for enroll=%s, cand=%s: %r",
                    enroll_str,
                    tmp_str,
                    exc,
                )
                continue

            score_val = float(score)
            logger.debug(
                "[Separator] Candidate %d -> score=%.4f, prediction=%s",
                idx,
                score_val,
                prediction,
            )

            #Using prediction to check is he the the speaker indeed
            if not bool(prediction):
                logger.debug(
                    "[Separator] Candidate %d rejected (prediction=False)", idx
                )
                continue

            if best_score is None or score_val > best_score:
                best_score = score_val
                best_tmp_path = tmp_path


        if best_tmp_path is None:
            logger.error("[Separator] No best candidate selected; check separation output.")
            raise RuntimeError("No best candidate selected; check separation output.")

        logger.info(
            "[Separator] Best candidate: %s with score=%.4f",
            best_tmp_path,
            best_score if best_score is not None else float("nan"),
        )

        # 3) Rename best candidate to the final output path
        try:
            if best_tmp_path != out_path:
                if out_path.exists():
                    logger.debug(
                        "[Separator] Output path already exists; removing: %s", out_path
                    )
                    out_path.unlink()
                best_tmp_path.rename(out_path)
        except Exception as exc:
            logger.exception(
                "[Separator] Failed to move best candidate %s -> %s: %r",
                best_tmp_path,
                out_path,
                exc,
            )
            raise

        logger.info("[Separator] Final purified WAV written to: %s", out_path)

        # 4) Clean up other temporary candidates
        for tmp in tmp_paths:
            if tmp.exists() and tmp != out_path:
                try:
                    tmp.unlink()
                    logger.debug("[Separator] Removed temporary candidate: %s", tmp)
                except Exception as exc:
                    logger.warning(
                        "[Separator] Failed to remove temporary file %s: %r",
                        tmp,
                        exc,
                    )
