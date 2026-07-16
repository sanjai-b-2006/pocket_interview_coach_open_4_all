import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def gen_id() -> str:
    return uuid.uuid4().hex


@dataclass
class Question:
    text: str
    sample_answer: str = ""
    persona: str = ""
    is_dynamic: bool = False
    id: str = field(default_factory=gen_id)
    answer: Optional["Answer"] = None


@dataclass
class Answer:
    transcript: str = ""
    duration_sec: float = 0.0
    words_per_minute: float = 0.0
    pitch_variation: float = 0.0
    filler_word_count: int = 0
    pause_ratio: float = 0.0
    volume_consistency: float = 0.0
    delivery_timeline: List[Dict[str, Any]] = field(default_factory=list)
    content_score: int = 0
    delivery_score: int = 0
    content_feedback: str = ""
    delivery_feedback: str = ""
    star_components: Dict[str, bool] = field(default_factory=dict)
    audio_bytes: Optional[bytes] = None
    improved_answer: str = ""  # independent-model rewrite of this answer (cached)


@dataclass
class InterviewSession:
    role: str
    company: str = ""
    experience_level: str = ""
    job_description: str = ""
    session_type: str = "job_interview"
    persona: str = ""
    panel_mode: bool = False
    drill_focus: str = ""
    questions: List[Question] = field(default_factory=list)
    id: str = field(default_factory=gen_id)
    summary: str = ""
    top_actions: List[str] = field(default_factory=list)
    cheat_sheet: str = ""
    hiring_verdict: Optional[Dict[str, Any]] = None  # judge-model result (cached)

    @property
    def answered_questions(self) -> List[Question]:
        return [q for q in self.questions if q.answer is not None]

    @property
    def avg_content_score(self) -> int:
        answered = self.answered_questions
        if not answered:
            return 0
        return round(sum(q.answer.content_score for q in answered) / len(answered))

    @property
    def avg_delivery_score(self) -> int:
        answered = self.answered_questions
        if not answered:
            return 0
        return round(sum(q.answer.delivery_score for q in answered) / len(answered))

    @property
    def readiness(self) -> Dict[str, Any]:
        score = round(self.avg_content_score * 0.6 + self.avg_delivery_score * 0.4)
        if score >= 90:
            grade = "A"
        elif score >= 80:
            grade = "B"
        elif score >= 70:
            grade = "C"
        elif score >= 60:
            grade = "D"
        else:
            grade = "F"
        return {"score": score, "grade": grade}
