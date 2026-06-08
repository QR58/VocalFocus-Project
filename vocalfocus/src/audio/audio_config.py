# src/audio/audio_config.py

import pyaudio
from pathlib import Path

# 1. Sample rate (Hz)
SAMPLE_RATE = 16000
#   16 kHz → matches most speech models

# 2. Channels
CHANNELS = 1
#   1 = mono (simplest, fastest - 1 mic)
#   2 = stereo (two mics or L/R - 2 mic)

# 3. Sample format
FORMAT = pyaudio.paInt16
#   16-bit PCM → wide support, good dynamic range

# 4. Sample width (bytes)
SAMPLE_WIDTH = 2  # paInt16 = 16 bits = 2 bytes

# 5. Chunk timing
CHUNK_MS = 100
#   frame length in milliseconds
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_MS / 1000)
#   samples per frame = rate × duration

# 6. STFT & Mel settings
WIN_LENGTH = int(0.025 * SAMPLE_RATE)
#   window = 25 ms
HOP_LENGTH = int(0.010 * SAMPLE_RATE)
#   hop = 10 ms
N_FFT = 512
#   FFT points ≥ window size
N_MELS = 40
#   Mel filter bands (20–80 common; 40 is a good default)

# ========================
# Project path structure
# ========================

# This file lives under src/audio/, so:
#   parents[0] -> src/audio
#   parents[1] -> src
#   parents[2] -> project root (vocalfocus/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RECORDINGS_ROOT = PROJECT_ROOT / "recordings"
DATA_ROOT = PROJECT_ROOT / "data"
DB_ROOT = PROJECT_ROOT / "db"

# Subfolders inside recordings/ (your structure stays clean)
ENROLL_RECORDINGS_DIR = RECORDINGS_ROOT / "enroll"      # enrollment voices
LIVE_MIX_DIR = RECORDINGS_ROOT / "mix"                  # raw live mixtures
PURIFIED_DIR = RECORDINGS_ROOT / "purified"             # target-only output

# Ensure folders exist (avoids “No such file or directory” when saving WAVs)
for d in (RECORDINGS_ROOT,ENROLL_RECORDINGS_DIR,LIVE_MIX_DIR,PURIFIED_DIR,DATA_ROOT,DB_ROOT):
    d.mkdir(parents=True, exist_ok=True)
