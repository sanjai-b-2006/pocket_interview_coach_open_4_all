import json
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from services.config import settings

BASIC_QUESTIONS = [
    "Tell me about yourself.",
    "What do you know about {company}, and why do you want to work here?",
    "Walk me through your resume.",
    "Why are you interested in this role?",
    "What makes you a good fit for this position?",
    "What are your thoughts on {company} and what we do?",
    "Why do you want to leave your current job?",
    "What would you say are your greatest strengths?",
    "What do you consider your biggest weakness?",
    "Where do you see yourself in five years?",
    "Why should we hire you over other candidates?",
    "What motivates you to do your best work?",
    "Describe your ideal work environment.",
    "How do you handle stress or working under pressure?",
    "What would you consider your greatest professional achievement?",
    "What are your salary expectations for this role?",
    "Why did you choose this career path?",
    "How would your previous manager or teammates describe you?",
    "Do you prefer working independently or as part of a team?",
    "How do you prioritize tasks when you have multiple deadlines?",
    "What's a significant challenge you've faced, and how did you overcome it?",
    "How do you respond to feedback or criticism?",
    "What are you looking for in your next role?",
    "Why did you decide to apply to {company}?",
    "What do you know about our industry and main competitors?",
    "What's something you're passionate about outside of work?",
    "How do you stay current in your field?",
]

PERSONAS = {
    "friendly_recruiter": (
        "a warm, encouraging recruiter who wants the candidate to succeed and helps them feel at "
        "ease, while still asking real questions"
    ),
    "tough_tech_lead": (
        "a demanding technical lead who probes deeply into technical decisions, pushes back on "
        "vague answers, and expects precision"
    ),
    "skeptical_panel": (
        "a skeptical panel interviewer who questions assumptions, looks for holes in reasoning, "
        "and rarely gives easy validation"
    ),
    "rapid_fire_founder": (
        "a fast-paced startup founder who asks rapid, direct questions and values concise, "
        "high-signal answers over long-winded ones"
    ),
}
PANEL_ROTATION = list(PERSONAS.keys())

SESSION_TYPES = {
    "job_interview": {
        "label": "Job Interview",
        "framing": "This is a standard job interview practice session.",
        "basic_questions": BASIC_QUESTIONS,
    },
    "salary_negotiation": {
        "label": "Salary Negotiation",
        "framing": (
            "This is a salary negotiation practice conversation, not a job interview -- focus on "
            "negotiation tactics, leverage, and framing, not general job qualifications."
        ),
        "basic_questions": [
            "What are your salary expectations for this role?",
            "Walk me through how you researched market rate for this position.",
            "What's most important to you in this offer besides base salary?",
            "Why do you believe you deserve this level of compensation?",
            "If we can't meet your number, what would you be willing to trade off?",
        ],
    },
    "performance_review": {
        "label": "Performance Review",
        "framing": "This is a performance review practice conversation between a manager and an employee.",
        "basic_questions": [
            "How do you think this review period went overall?",
            "What accomplishment are you most proud of this period?",
            "Where do you think you could have done better?",
            "How would you rate your own performance, and why?",
            "What support do you need from me to grow in the next period?",
        ],
    },
    "difficult_feedback": {
        "label": "Difficult Feedback Conversation",
        "framing": "This is practice for delivering or receiving difficult professional feedback.",
        "basic_questions": [
            "Can you walk me through what happened from your perspective?",
            "How did you feel about how that situation was handled?",
            "What would you do differently if this came up again?",
            "What impact do you think this had on the team?",
            "What do you need from me to move forward productively?",
        ],
    },
}


class GemmaAPIError(Exception):
    """Raised when the LLM call fails or returns an unusable response."""


@dataclass
class LLMOverride:
    """Per-request bring-your-own-key override. Any unset field falls back to server defaults."""

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None


def _extract_json(text: str) -> Dict[str, Any]:
    """Gemma sometimes wraps JSON in prose or code fences; pull the first {...} block out."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise GemmaAPIError(f"No JSON object found in model response: {text!r}")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise GemmaAPIError(f"Model response was not valid JSON: {exc}") from exc


class GemmaClient:
    def _chat(
        self,
        system: str,
        user: str,
        override: Optional[LLMOverride] = None,
        max_tokens: int = 700,
    ) -> str:
        api_key = (override and override.api_key) or settings.api_key
        base_url = (override and override.base_url) or settings.base_url
        model = (override and override.model) or settings.coach_model

        try:
            resp = httpx.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": 0.4,
                    "max_tokens": max_tokens,
                },
                timeout=45.0,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise GemmaAPIError(
                f"LLM API returned {exc.response.status_code}: {exc.response.text[:300]}"
            ) from exc
        except httpx.RequestError as exc:
            raise GemmaAPIError(f"Could not reach LLM API: {exc}") from exc

        try:
            return resp.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as exc:
            raise GemmaAPIError(f"Unexpected LLM response shape: {exc}") from exc

    def generate_questions(
        self,
        role: str,
        job_description: str,
        num_questions: int,
        company: str = "",
        experience_level: str = "",
        resume_text: str = "",
        session_type: str = "job_interview",
        persona: str = "",
        panel_mode: bool = False,
        drill_focus: str = "",
        override: Optional[LLMOverride] = None,
    ) -> List[Dict[str, str]]:
        type_config = SESSION_TYPES.get(session_type, SESSION_TYPES["job_interview"])
        basic_pool = type_config["basic_questions"]
        framing = type_config["framing"]

        basic_question = None
        remaining = num_questions
        if not drill_focus:
            basic_question = random.choice(basic_pool).format(company=company or "the company")
            remaining = max(num_questions - 1, 0)

        if panel_mode:
            persona_instruction = (
                "This is a PANEL session with multiple interviewers, each with a distinct voice: "
                + "; ".join(f'"{key}" = {desc}' for key, desc in PERSONAS.items())
                + '. For EVERY question in your response, include a "persona" field set to exactly '
                "one of: " + ", ".join(f'"{k}"' for k in PERSONAS.keys()) + ", rotating through "
                "them roughly evenly across the questions."
            )
        elif persona and persona in PERSONAS:
            persona_instruction = (
                f"Write every question and sample answer in the voice of this interviewer "
                f"persona: {PERSONAS[persona]}. Set every question's \"persona\" field to "
                f'"{persona}".'
            )
        else:
            persona_instruction = 'Set every question\'s "persona" field to "".'

        drill_instruction = (
            f"This is a focused practice DRILL, not a full session -- every one of the "
            f"{num_questions} questions must specifically target and exercise this weak area: "
            f'"{drill_focus}". Make each question a fresh angle for practicing that exact skill.\n'
            if drill_focus
            else ""
        )

        system = (
            f"You are an expert interview coach. {framing} Respond ONLY with a JSON object: "
            '{"questions": [{"text": "...", "sample_answer": "...", "persona": "..."}, ...]}. '
            "sample_answer should be a strong, concise model answer (2-3 short sentences, no more) "
            "a top candidate might give, useful for the candidate to compare against afterward. "
            "Keep every question's text to one sentence. Be concise everywhere to respond quickly. "
            f"{persona_instruction}"
        )
        experience_line = (
            f"Candidate's experience level: {experience_level}. Calibrate question difficulty and "
            "depth to this level.\n"
            if experience_level.strip()
            else ""
        )
        resume_block = (
            "Candidate's resume (use it to ask at least one specific question about a real "
            f"project or piece of experience mentioned in it):\n{resume_text}\n"
            if resume_text.strip()
            else ""
        )

        if drill_focus:
            user = (
                f"{drill_instruction}"
                f"Role: {role}"
                + (f" at {company}" if company else "")
                + f".\nJob description context (may be empty): {job_description}\n"
                f"{experience_line}{resume_block}"
                f"Return exactly {num_questions} drill questions total, ordered easiest to hardest."
            )
        else:
            user = (
                f"The FIRST question must be exactly this basic warm-up question (do not alter the "
                f'wording): "{basic_question}" -- write a tailored sample_answer for it given the '
                f"role/company context below.\n"
                f"{experience_line}"
                f"Then generate {remaining} additional questions for the role: {role}"
                + (f" at {company}" if company else "")
                + f".\nJob description context (may be empty): {job_description}\n"
                f"{resume_block}"
                "Mix behavioral questions with role-specific technical questions tailored to this "
                "role's actual industry/domain (e.g. a manufacturing role should get questions "
                "about topics like quality control or product recalls, not generic software "
                "questions). Order the additional questions from EASIEST to HARDEST -- the last "
                "question should be the most challenging. "
                f"Return {num_questions} questions total, in order, with the basic warm-up "
                "question first."
            )

        content = self._chat(system, user, override, max_tokens=min(1800, 250 + num_questions * 110))
        data = _extract_json(content)
        questions = [
            {
                "text": q["text"],
                "sample_answer": q.get("sample_answer", ""),
                "persona": q.get("persona", "") if (panel_mode or persona) else "",
            }
            for q in list(data["questions"])[:num_questions]
        ]
        if not drill_focus and (not questions or questions[0]["text"].strip() != basic_question):
            questions.insert(
                0, {"text": basic_question, "sample_answer": "", "persona": persona if persona and not panel_mode else ""}
            )
            questions = questions[:num_questions]
        return questions

    def generate_answer_feedback(
        self,
        question: str,
        transcript: str,
        prosody: Dict[str, float],
        persona: str = "",
        override: Optional[LLMOverride] = None,
    ) -> Dict[str, Any]:
        persona_desc = PERSONAS.get(persona, "a skilled, perceptive interviewer")
        system = (
            "You are a supportive but honest interview coach. You are given a candidate's "
            "transcribed answer plus objective speech-delivery measurements. "
            "Respond ONLY with a JSON object with keys: "
            "content_score (0-100 int), delivery_score (0-100 int), "
            "content_feedback (string, 1-3 sentences on structure/specificity/STAR method), "
            "delivery_feedback (string, 1-3 sentences translating the raw speech metrics into "
            "plain-English coaching about pace, tone/monotone, filler words, and pauses), "
            "star_components (object with boolean keys situation, task, action, result -- true if "
            "the answer clearly included that STAR element, false if it was missing or too vague; "
            "only meaningful for behavioral/story-style questions, otherwise set all to false), "
            "follow_up (string or null: a natural, challenging follow-up question a real "
            f"interviewer -- specifically {persona_desc} -- would ask to probe deeper or play "
            "devil's advocate if this answer left something unexplored or unconvincing; null if "
            "the answer was already thorough)."
        )
        user = (
            f"Question: {question}\n"
            f"Transcript: {transcript}\n"
            f"Speech metrics: words_per_minute={prosody['words_per_minute']:.0f}, "
            f"pitch_variation={prosody['pitch_variation']:.2f} (0=monotone, 1=highly expressive), "
            f"filler_word_count={prosody['filler_word_count']}, "
            f"pause_ratio={prosody['pause_ratio']:.2f} (fraction of time silent), "
            f"volume_consistency={prosody['volume_consistency']:.2f} (1=very steady)."
        )
        content = self._chat(system, user, override, max_tokens=500)
        return _extract_json(content)

    def generate_report(
        self,
        role: str,
        qa_pairs: List[Dict[str, Any]],
        avg_content_score: int,
        avg_delivery_score: int,
        session_type: str = "job_interview",
        override: Optional[LLMOverride] = None,
    ) -> Dict[str, Any]:
        framing = SESSION_TYPES.get(session_type, SESSION_TYPES["job_interview"])["framing"]
        system = (
            f"You are a coach writing a short end-of-session report. {framing} "
            "Respond ONLY with a JSON object: "
            '{"summary": "2-4 sentence overview", "top_actions": ["action 1", "action 2", "action 3"]}.'
        )
        transcript_block = "\n".join(
            f"- Q: {qa['question']}\n  A: {qa['transcript']}\n"
            f"  content_score={qa['content_score']} delivery_score={qa['delivery_score']}"
            for qa in qa_pairs
        )
        user = (
            f"Role: {role}\n"
            f"Average content score: {avg_content_score}/100\n"
            f"Average delivery score: {avg_delivery_score}/100\n"
            f"Per-question detail:\n{transcript_block}\n"
            "Write an encouraging but candid summary and 3 concrete, prioritized next actions."
        )
        content = self._chat(system, user, override, max_tokens=400)
        return _extract_json(content)

    def generate_cheat_sheet(
        self,
        role: str,
        qa_pairs: List[Dict[str, Any]],
        override: Optional[LLMOverride] = None,
    ) -> str:
        system = (
            "You are an interview coach creating a personalized one-page 'cheat sheet' for a "
            "candidate to glance at right before their real interview. Respond ONLY with a JSON "
            'object: {"cheat_sheet": "markdown text"}. Use short bullet points grouped under 2-4 '
            "headings (e.g. Key Stories, Strengths to Emphasize, Watch-outs). Base it on the "
            "candidate's actual best-scoring answers below -- distill their real talking points, "
            "don't invent generic advice."
        )
        transcript_block = "\n".join(
            f"- Q: {qa['question']}\n  A: {qa['transcript']}\n"
            f"  content_score={qa['content_score']} delivery_score={qa['delivery_score']}"
            for qa in qa_pairs
        )
        user = f"Role: {role}\nPer-question detail:\n{transcript_block}\n"
        content = self._chat(system, user, override, max_tokens=600)
        return _extract_json(content).get("cheat_sheet", "")


gemma_client = GemmaClient()
