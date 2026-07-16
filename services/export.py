import json

from fpdf import FPDF

from services.models import InterviewSession


def _latin1(text: str) -> str:
    """fpdf2's core fonts only support latin-1; strip anything else rather than crash."""
    return text.encode("latin-1", "replace").decode("latin-1")


def _line(pdf: FPDF, text: str, size: int = 11, style: str = "") -> None:
    """multi_cell handles width=0 and cursor reset far more reliably than cell(..., ln=True)."""
    pdf.set_font("Helvetica", style, size)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 6 if size <= 11 else 8, _latin1(text))


def generate_report_pdf(session: InterviewSession) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    _line(pdf, "Pocket Interview Coach - Session Report", size=18, style="B")
    label = f"{session.role} @ {session.company}" if session.company else session.role
    _line(pdf, label)
    readiness = session.readiness
    _line(pdf, f"Readiness score: {readiness['score']}/100 (Grade {readiness['grade']})")
    _line(pdf, f"Avg content: {session.avg_content_score}/100  |  Avg delivery: {session.avg_delivery_score}/100")
    pdf.ln(4)

    if session.hiring_verdict:
        v = session.hiring_verdict
        _line(pdf, "Independent Hiring Verdict (second model)", size=13, style="B")
        _line(pdf, f"Decision: {v.get('decision', '-')}  |  Confidence: {v.get('confidence', '-')}%")
        if v.get("rationale"):
            _line(pdf, v["rationale"])
        if v.get("case_for"):
            _line(pdf, f"Case for: {v['case_for']}")
        if v.get("case_against"):
            _line(pdf, f"Biggest concern: {v['case_against']}")
        pdf.ln(2)

    _line(pdf, "Summary", size=13, style="B")
    _line(pdf, session.summary or "")
    pdf.ln(2)

    _line(pdf, "Top Actions", size=13, style="B")
    for action in session.top_actions:
        _line(pdf, f"- {action}")
    pdf.ln(2)

    _line(pdf, "Per-Question Detail", size=13, style="B")
    for i, q in enumerate(session.answered_questions):
        _line(pdf, f"Q{i + 1}: {q.text}", size=11, style="B")
        _line(pdf, f'"{q.answer.transcript}"', size=10, style="I")
        _line(
            pdf,
            f"Content {q.answer.content_score}/100, Delivery {q.answer.delivery_score}/100 -- "
            f"{q.answer.content_feedback} {q.answer.delivery_feedback}",
            size=10,
        )
        pdf.ln(3)

    return bytes(pdf.output())


def generate_transcript_text(session: InterviewSession) -> str:
    lines = [f"Pocket Interview Coach -- {session.role}"]
    if session.company:
        lines[0] += f" @ {session.company}"
    lines.append("")
    for i, q in enumerate(session.answered_questions):
        lines.append(f"Q{i + 1}: {q.text}")
        lines.append(f"A: {q.answer.transcript}")
        lines.append(f"  content_score={q.answer.content_score} delivery_score={q.answer.delivery_score}")
        lines.append("")
    return "\n".join(lines)


def generate_report_json(session: InterviewSession) -> str:
    data = {
        "role": session.role,
        "company": session.company,
        "session_type": session.session_type,
        "avg_content_score": session.avg_content_score,
        "avg_delivery_score": session.avg_delivery_score,
        "readiness": session.readiness,
        "hiring_verdict": session.hiring_verdict,
        "summary": session.summary,
        "top_actions": session.top_actions,
        "answers": [
            {
                "question": q.text,
                "transcript": q.answer.transcript,
                "content_score": q.answer.content_score,
                "delivery_score": q.answer.delivery_score,
                "content_feedback": q.answer.content_feedback,
                "delivery_feedback": q.answer.delivery_feedback,
                "star_components": q.answer.star_components,
                "improved_answer": q.answer.improved_answer,
                "words_per_minute": q.answer.words_per_minute,
                "filler_word_count": q.answer.filler_word_count,
            }
            for q in session.answered_questions
        ],
    }
    return json.dumps(data, indent=2)
