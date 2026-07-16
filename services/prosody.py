import math
import re
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np

from services.config import settings
from services.asr import TranscriptResult

FILLER_WORDS = {"um", "umm", "uh", "uhh", "like", "actually", "basically", "literally"}

_PAUSE_THRESHOLD_SEC = 0.4
_TIMELINE_SEGMENT_SECONDS = 5.0
_TIMELINE_MAX_SEGMENTS = 8


def _words_per_minute(word_count: int, duration_sec: float) -> float:
    if duration_sec <= 0:
        return 0.0
    return word_count / (duration_sec / 60.0)


def _filler_word_count(transcript_text: str) -> int:
    tokens = re.findall(r"[a-zA-Z']+", transcript_text.lower())
    return sum(1 for t in tokens if t in FILLER_WORDS)


def _pause_ratio(words: List, duration_sec: float) -> float:
    if duration_sec <= 0 or len(words) < 2:
        return 0.0
    gap_total = 0.0
    for prev, curr in zip(words, words[1:]):
        gap = curr.start - prev.end
        if gap > _PAUSE_THRESHOLD_SEC:
            gap_total += gap
    return float(min(gap_total / duration_sec, 1.0))


def _compute_f0(
    y: np.ndarray, sr: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Runs pitch tracking once so both the aggregate metric and the per-segment
    timeline can reuse it, instead of re-running the expensive pyin pass repeatedly."""
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
        )
    except Exception:
        return None, None, None
    times = librosa.times_like(f0, sr=sr)
    return f0, voiced_flag, times


def _pitch_variation_from_f0(f0: Optional[np.ndarray], voiced_flag: Optional[np.ndarray]) -> float:
    if f0 is None or f0.size == 0:
        return 0.0
    voiced = f0[voiced_flag] if voiced_flag is not None else f0[~np.isnan(f0)]
    voiced = voiced[~np.isnan(voiced)]
    if voiced.size < 2 or voiced.mean() == 0:
        return 0.0
    coeff_of_variation = float(np.std(voiced) / np.mean(voiced))
    # Typical conversational speech CoV ~0.05-0.25; normalize and clip to 0-1.
    return float(np.clip(coeff_of_variation / 0.25, 0.0, 1.0))


def _volume_consistency(y: np.ndarray) -> float:
    rms = librosa.feature.rms(y=y)[0]
    if rms.size == 0 or rms.mean() == 0:
        return 0.0
    coeff_of_variation = float(np.std(rms) / np.mean(rms))
    # Lower variation = steadier volume; invert so 1.0 means very steady.
    return float(np.clip(1.0 - coeff_of_variation, 0.0, 1.0))


def _delivery_timeline(
    words: List,
    duration_sec: float,
    f0: Optional[np.ndarray],
    voiced_flag: Optional[np.ndarray],
    times: Optional[np.ndarray],
) -> List[Dict[str, float]]:
    if duration_sec <= 0:
        return []
    num_segments = max(1, min(_TIMELINE_MAX_SEGMENTS, math.ceil(duration_sec / _TIMELINE_SEGMENT_SECONDS)))
    segment_len = duration_sec / num_segments

    timeline = []
    for i in range(num_segments):
        t0, t1 = i * segment_len, (i + 1) * segment_len
        seg_words = [w for w in words if t0 <= w.start < t1]
        seg_wpm = _words_per_minute(len(seg_words), segment_len)

        seg_pitch = 0.0
        if f0 is not None and times is not None:
            mask = (times >= t0) & (times < t1)
            seg_f0 = f0[mask]
            seg_voiced = voiced_flag[mask] if voiced_flag is not None else None
            seg_pitch = _pitch_variation_from_f0(seg_f0, seg_voiced)

        timeline.append({"t": round(t0, 1), "words_per_minute": round(seg_wpm, 1), "pitch_variation": round(seg_pitch, 3)})
    return timeline


def compute_prosody(audio_path: str, transcript: TranscriptResult) -> Dict[str, object]:
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    word_count = len(transcript.words) if transcript.words else len(transcript.text.split())

    # pyin's numba JIT compilation is a real memory/CPU spike -- skip it on constrained hosts.
    f0, voiced_flag, times = _compute_f0(y, sr) if settings.enable_pitch_analysis else (None, None, None)

    return {
        "words_per_minute": _words_per_minute(word_count, transcript.duration_sec),
        "pitch_variation": _pitch_variation_from_f0(f0, voiced_flag),
        "filler_word_count": _filler_word_count(transcript.text),
        "pause_ratio": _pause_ratio(transcript.words, transcript.duration_sec),
        "volume_consistency": _volume_consistency(y),
        "delivery_timeline": _delivery_timeline(transcript.words, transcript.duration_sec, f0, voiced_flag, times),
    }
