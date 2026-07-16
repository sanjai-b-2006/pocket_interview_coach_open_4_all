import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Everything runs through OpenRouter (or any OpenAI-compatible endpoint), so a single API key
    # powers the whole app — no provider-specific setup needed. This is the "open for all" build.
    api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    base_url: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    # Coach model — writes questions, scores answers, asks follow-ups, builds the cheat sheet.
    coach_model: str = os.getenv("COACH_MODEL", os.getenv("GEMMA_MODEL", "google/gemma-4-26b-a4b-it"))

    # Independent "hiring manager" — a DIFFERENT model, for a genuine second opinion rather than
    # the coach grading itself. Same OpenRouter key/endpoint by default; only the model differs.
    judge_api_key: str = os.getenv("JUDGE_API_KEY", "") or api_key
    judge_base_url: str = os.getenv("JUDGE_BASE_URL", "") or base_url
    judge_model: str = os.getenv("JUDGE_MODEL", "openai/gpt-4o-mini")

    asr_device: str = os.getenv("ASR_DEVICE", "cpu")
    asr_model_size: str = os.getenv("ASR_MODEL_SIZE", "base")
    asr_compute_type: str = os.getenv("ASR_COMPUTE_TYPE", "int8")
    enable_pitch_analysis: bool = os.getenv("ENABLE_PITCH_ANALYSIS", "true").lower() in ("1", "true", "yes")


settings = Settings()
