from typing import Optional, Tuple

from services import asr, prosody
from services.llm import LLMOverride, PANEL_ROTATION, gemma_client
from services.models import Answer, InterviewSession, Question


def create_session(
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
) -> InterviewSession:
    questions_data = gemma_client.generate_questions(
        role,
        job_description,
        num_questions,
        company=company,
        experience_level=experience_level,
        resume_text=resume_text,
        session_type=session_type,
        persona=persona,
        panel_mode=panel_mode,
        drill_focus=drill_focus,
        override=override,
    )

    session = InterviewSession(
        role=role,
        company=company,
        experience_level=experience_level,
        job_description=job_description,
        session_type=session_type,
        persona=persona,
        panel_mode=panel_mode,
        drill_focus=drill_focus,
    )

    for i, q in enumerate(questions_data):
        persona_for_q = q.get("persona") or (PANEL_ROTATION[i % len(PANEL_ROTATION)] if panel_mode else persona)
        session.questions.append(
            Question(text=q["text"], sample_answer=q["sample_answer"], persona=persona_for_q)
        )

    return session


def process_answer(
    session: InterviewSession,
    question: Question,
    audio_path: str,
    override: Optional[LLMOverride] = None,
    audio_bytes: Optional[bytes] = None,
) -> Tuple[Answer, Optional[Question]]:
    transcript = asr.transcribe(audio_path)
    features = prosody.compute_prosody(audio_path, transcript)

    effective_persona = question.persona or session.persona

    feedback = gemma_client.generate_answer_feedback(
        question=question.text,
        transcript=transcript.text,
        prosody=features,
        persona=effective_persona,
        override=override,
    )

    answer = Answer(
        transcript=transcript.text,
        duration_sec=transcript.duration_sec,
        words_per_minute=features["words_per_minute"],
        pitch_variation=features["pitch_variation"],
        filler_word_count=features["filler_word_count"],
        pause_ratio=features["pause_ratio"],
        volume_consistency=features["volume_consistency"],
        delivery_timeline=features.get("delivery_timeline", []),
        content_score=int(feedback["content_score"]),
        delivery_score=int(feedback["delivery_score"]),
        content_feedback=feedback["content_feedback"],
        delivery_feedback=feedback["delivery_feedback"],
        star_components=feedback.get("star_components", {}) or {},
        audio_bytes=audio_bytes,
    )
    question.answer = answer

    follow_up_question: Optional[Question] = None
    follow_up_text = feedback.get("follow_up")
    if follow_up_text:
        follow_up_question = Question(
            text=follow_up_text,
            sample_answer="",
            persona=effective_persona,
            is_dynamic=True,
        )
        idx = session.questions.index(question)
        session.questions.insert(idx + 1, follow_up_question)

    return answer, follow_up_question


def build_report(session: InterviewSession, override: Optional[LLMOverride] = None) -> InterviewSession:
    answered = session.answered_questions
    if not answered:
        raise ValueError("Cannot build a report for a session with no answered questions")

    qa_pairs = [
        {
            "question": q.text,
            "transcript": q.answer.transcript,
            "content_score": q.answer.content_score,
            "delivery_score": q.answer.delivery_score,
        }
        for q in answered
    ]

    llm_report = gemma_client.generate_report(
        role=session.role,
        qa_pairs=qa_pairs,
        avg_content_score=session.avg_content_score,
        avg_delivery_score=session.avg_delivery_score,
        session_type=session.session_type,
        override=override,
    )
    session.summary = llm_report["summary"]
    session.top_actions = list(llm_report["top_actions"])
    return session


def build_cheat_sheet(session: InterviewSession, override: Optional[LLMOverride] = None) -> str:
    if session.cheat_sheet:
        return session.cheat_sheet

    qa_pairs = [
        {
            "question": q.text,
            "transcript": q.answer.transcript,
            "content_score": q.answer.content_score,
            "delivery_score": q.answer.delivery_score,
        }
        for q in session.answered_questions
    ]
    session.cheat_sheet = gemma_client.generate_cheat_sheet(role=session.role, qa_pairs=qa_pairs, override=override)
    return session.cheat_sheet
