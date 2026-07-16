"""Independent 'hiring manager' powered by a second, different model.

This is deliberately a *different* model from the coach in ``llm.py``. The coach coaches the
candidate; this model plays the skeptical hiring manager who has to actually make a call. Using two
independent models means the final verdict is a genuine second opinion rather than the same model
grading its own coaching. Both run through OpenRouter by default (a single API key), differing only
in which model is used.
"""

import json
import re
from typing import Any, Dict

import httpx

from services.config import settings
from services.models import InterviewSession


class JudgeError(Exception):
    """Raised when the judge model call fails or returns an unusable response."""


DECISION_ORDER = ["Strong No-Hire", "No-Hire", "Lean No-Hire", "Lean Hire", "Hire", "Strong Hire"]


def _extract_json(text: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise JudgeError(f"No JSON object found in judge response: {text!r}")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise JudgeError(f"Judge response was not valid JSON: {exc}") from exc


class JudgeClient:
    def is_configured(self) -> bool:
        return bool(settings.judge_api_key)

    def _chat(self, system: str, user: str, max_tokens: int = 1200) -> str:
        if not settings.judge_api_key:
            raise JudgeError(
                "No judge API key configured. Set JUDGE_API_KEY (or OPENROUTER_API_KEY) to enable "
                "the independent hiring verdict."
            )
        try:
            resp = httpx.post(
                f"{settings.judge_base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {settings.judge_api_key}"},
                json={
                    "model": settings.judge_model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": 0.3,
                    "max_tokens": max_tokens,
                },
                timeout=60.0,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise JudgeError(
                f"Judge API returned {exc.response.status_code}: {exc.response.text[:300]}"
            ) from exc
        except httpx.RequestError as exc:
            raise JudgeError(f"Could not reach the judge API: {exc}") from exc

        try:
            choice = resp.json()["choices"][0]
            content = choice["message"].get("content") or ""
        except (KeyError, IndexError, ValueError) as exc:
            raise JudgeError(f"Unexpected judge response shape: {exc}") from exc
        # Some (reasoning) models can burn the whole token budget on hidden reasoning and return an
        # empty `content`. Surface that clearly instead of a downstream JSON parse error.
        if not content.strip():
            raise JudgeError(
                f"Judge returned empty content (finish_reason="
                f"{choice.get('finish_reason')!r}); likely truncated -- try again."
            )
        return content

    def hiring_verdict(self, session: InterviewSession) -> Dict[str, Any]:
        """Have the judge model make an actual hire/no-hire call on the whole session."""
        transcript_block = "\n".join(
            f"- Q: {q.text}\n  A: {q.answer.transcript}\n"
            f"  (coach scored content {q.answer.content_score}/100, delivery {q.answer.delivery_score}/100)"
            for q in session.answered_questions
        )
        label = f"{session.role}" + (f" at {session.company}" if session.company else "")
        system = (
            "You are a seasoned hiring manager making a real call after an interview. Be fair but "
            "decisive -- you must commit to a recommendation, not sit on the fence. Respond ONLY "
            "with a JSON object with keys: "
            'decision (one of exactly: "Strong No-Hire", "No-Hire", "Lean No-Hire", "Lean Hire", '
            '"Hire", "Strong Hire"), '
            "confidence (int 0-100, how sure you are of this call), "
            "rationale (2-3 sentence explanation in the first person, as the hiring manager), "
            "case_for (one sentence: the single strongest reason to hire this candidate), "
            "case_against (one sentence: the single biggest concern or gap), "
            "standout_moment (one short sentence quoting or paraphrasing the best thing they said, "
            "or empty string if nothing stood out)."
        )
        user = (
            f"Role being interviewed for: {label}\n"
            f"Overall coach scores: content {session.avg_content_score}/100, "
            f"delivery {session.avg_delivery_score}/100.\n"
            f"Full interview:\n{transcript_block}\n\n"
            "Make your hiring recommendation now."
        )
        data = _extract_json(self._chat(system, user, max_tokens=1400))
        # Models sometimes emit Unicode dashes (non-breaking hyphen / en-dash) instead of ASCII
        # "-", which would silently break the decision + color lookup. Normalize before matching.
        decision = str(data.get("decision", "")).replace("‑", "-").replace("–", "-").strip()
        if decision not in DECISION_ORDER:
            # Snap anything unexpected to the nearest sensible bucket instead of crashing the UI.
            decision = "Lean Hire"
        data["decision"] = decision
        return data

    def rewrite_answer(self, question: str, transcript: str, role: str) -> str:
        """Rewrite the candidate's actual answer into a stronger version of *their own* answer."""
        system = (
            "You are an elite interview coach. Rewrite the candidate's answer into a stronger "
            "version that a top performer for this role would give. Keep it grounded in what they "
            "actually said -- sharpen structure (ideally STAR), add specificity and a concrete "
            "result, and cut filler -- but do NOT invent fake achievements or numbers they didn't "
            "mention. Keep it to 3-5 sentences, spoken-word style (this is said out loud, not "
            "written). Respond ONLY with a JSON object: {\"improved_answer\": \"...\"}."
        )
        user = f"Role: {role}\nQuestion: {question}\nCandidate's actual answer: {transcript}\n"
        data = _extract_json(self._chat(system, user, max_tokens=1400))
        return data.get("improved_answer", "").strip()


judge_client = JudgeClient()
