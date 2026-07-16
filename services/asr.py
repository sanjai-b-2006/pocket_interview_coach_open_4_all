from dataclasses import dataclass, field
from typing import List

from faster_whisper import WhisperModel

from services.config import settings

_model: WhisperModel | None = None


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        # CPU now (dev machine has no ROCm GPU). Swap ASR_DEVICE=cuda once running on
        # AMD Developer Cloud with ROCm-enabled torch/ctranslate2 -- no code change needed.
        _model = WhisperModel(
            settings.asr_model_size,
            device=settings.asr_device,
            compute_type=settings.asr_compute_type,
        )
    return _model


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class TranscriptResult:
    text: str
    duration_sec: float
    words: List[Word] = field(default_factory=list)


def transcribe(audio_path: str) -> TranscriptResult:
    model = _get_model()
    segments, info = model.transcribe(audio_path, word_timestamps=True)

    words: List[Word] = []
    text_parts: List[str] = []
    for segment in segments:
        text_parts.append(segment.text)
        if segment.words:
            for w in segment.words:
                words.append(Word(text=w.word.strip(), start=w.start, end=w.end))

    return TranscriptResult(
        text=" ".join(part.strip() for part in text_parts).strip(),
        duration_sec=info.duration,
        words=words,
    )
