from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional
import json

from audio import audio_config

# profiles.json will live under db/
PROFILES_PATH = audio_config.DB_ROOT / "profiles.json"


@dataclass
class Profile:
    name: str
    enroll_wav: str  # absolute path to enrollment WAV


def _load_profiles_raw() -> List[dict]:
    if not PROFILES_PATH.exists():
        return []
    try:
        with PROFILES_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # If corrupted, start from empty list
        return []


def _save_profiles_raw(items: List[dict]) -> None:
    with PROFILES_PATH.open("w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)


def load_profiles() -> List[Profile]:
    items = _load_profiles_raw()
    return [
        Profile(**item)
        for item in items
        if "name" in item and "enroll_wav" in item
    ]


def save_profiles(profiles: List[Profile]) -> None:
    items = [asdict(p) for p in profiles]
    _save_profiles_raw(items)


def create_profile(*, name: str, enroll_wav_path: str) -> Profile:
    """
    Create or update a profile with the given name and enrollment WAV path.
    """
    path = Path(enroll_wav_path).resolve()
    profiles = load_profiles()

    # Remove existing profile with same name (if any)
    profiles = [p for p in profiles if p.name != name]

    profile = Profile(name=name, enroll_wav=str(path))
    profiles.append(profile)
    save_profiles(profiles)
    return profile


def get_profile_by_name(name: str) -> Optional[Profile]:
    for p in load_profiles():
        if p.name == name:
            return p
    return None