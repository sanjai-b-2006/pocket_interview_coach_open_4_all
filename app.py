import os
import random
import re
import tempfile

import plotly.graph_objects as go
import streamlit as st
from streamlit_mic_recorder import mic_recorder

from services import export as export_service
from services import interview
from services import resume as resume_service
from services.judge import JudgeError, judge_client
from services.llm import LLMOverride, PERSONAS, SESSION_TYPES
from services.models import InterviewSession, Question

st.set_page_config(page_title="Pocket Interview Coach", page_icon="🎤", layout="centered")

ACCENT = "#9b6bff"
ACCENT_2 = "#2dd4ee"
ACCENT_3 = "#ff8a5c"

PRESET_ROLES = ["Software Engineer", "Product Manager", "Data Analyst"]
SURPRISE_ROLES = [
    ("UX Designer", "Figma"),
    ("Data Scientist", "Netflix"),
    ("DevOps Engineer", "Stripe"),
    ("Marketing Manager", "Notion"),
    ("Solutions Architect", "AMD"),
    ("Mechanical Engineer", "Tesla"),
    ("Nurse Practitioner", "Mayo Clinic"),
    ("High School Teacher", ""),
    ("Restaurant General Manager", ""),
    ("Venture Capital Associate", "Sequoia"),
]
EXPERIENCE_LEVELS = ["Entry-level (0-1 yrs)", "Mid-level (2-5 yrs)", "Senior (5+ yrs)", "Staff/Lead (8+ yrs)"]
SAMPLE_ANSWER_MODES = {"After each answer": "after", "At the end": "end", "Don't show": "off"}
STAR_LABELS = {"situation": "Situation", "task": "Task", "action": "Action", "result": "Result"}

VERDICT_COLORS = {
    "Strong Hire": "#27c98f",
    "Hire": "#3fae6d",
    "Lean Hire": "#8bb84a",
    "Lean No-Hire": "#e0a63c",
    "No-Hire": "#e07a3c",
    "Strong No-Hire": "#e0563c",
}

# Boilerplate/stopwords stripped out before computing job-description keyword coverage.
_STOPWORDS = set(
    "the a an and or but for with without to of in on at by from as is are be been being this that "
    "these those you your we our us they their it its will would should can could may might must have "
    "has had do does did not no yes if then than so such about into over under more most other some "
    "any all each both few many who whom which what when where why how work working role team teams "
    "experience years year strong ability able skills skill required requirements responsibilities "
    "including etc across using use used within per via new plus etc.".split()
)

st.markdown(
    f"""
    <style>
    h1, h2, h3 {{ font-weight: 700; }}
    .gradient-text {{
        background: linear-gradient(90deg, {ACCENT_2}, {ACCENT});
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }}
    .persona-badge {{
        display: inline-block;
        border: 1px solid #292942;
        border-radius: 999px;
        padding: 2px 12px;
        font-size: 0.75rem;
        color: #cfcfe6;
        background: rgba(255,255,255,0.04);
    }}
    .qtag {{
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        padding: 3px 12px;
        border-radius: 999px;
        text-transform: uppercase;
    }}
    .qtag-question {{ color: {ACCENT_2}; background: rgba(45,212,238,0.12); border: 1px solid rgba(45,212,238,0.4); }}
    .qtag-followup {{ color: {ACCENT_3}; background: rgba(255,138,92,0.12); border: 1px solid rgba(255,138,92,0.45); }}
    .question-text {{
        font-size: 1.9rem;
        line-height: 1.3;
        font-weight: 700;
        margin: 14px 0 6px 0;
        color: #f7f7fb;
    }}
    .answer-hint {{ color: #8888a0; font-size: 0.9rem; margin: 2px 0 8px 0; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state():
    defaults = {
        "page": "setup",
        "session": None,
        "current_index": 0,
        "current_feedback": None,
        "sample_answer_mode": "end",
        "byok_api_key": "",
        "byok_base_url": "",
        "byok_model": "",
        "drill_prefill": None,
        "timer_enabled": False,
        "timer_seconds": 90,
        "surprise_pick": None,
        "attempts": {},  # question.id -> recording attempt number (bumped to reset the recorder)
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_override() -> LLMOverride:
    return LLMOverride(
        api_key=st.session_state.byok_api_key or None,
        base_url=st.session_state.byok_base_url or None,
        model=st.session_state.byok_model or None,
    )


def render_sidebar():
    with st.sidebar:
        st.markdown("### 🎤 Pocket Interview Coach")
        if st.session_state.page != "setup":
            if st.button("🏠 Start new interview", use_container_width=True):
                st.session_state.page = "setup"
                st.session_state.session = None
                st.session_state.current_index = 0
                st.session_state.current_feedback = None
                st.rerun()
        with st.expander("⚙️ Bring your own API key"):
            st.caption("Leave blank to use the app's default key/model.")
            st.session_state.byok_api_key = st.text_input(
                "API key", value=st.session_state.byok_api_key, type="password"
            )
            st.session_state.byok_base_url = st.text_input(
                "Base URL", value=st.session_state.byok_base_url, placeholder="https://openrouter.ai/api/v1"
            )
            st.session_state.byok_model = st.text_input(
                "Model", value=st.session_state.byok_model, placeholder="google/gemma-4-26b-a4b-it"
            )
        st.markdown("---")
        judge_on = judge_client.is_configured()
        st.caption(
            "🧠 **Coach:** Gemma 4\n\n"
            + ("🔥 **Hiring manager:** independent model (OpenRouter)" if judge_on else "🔥 Verdict: _add JUDGE_API_KEY_")
        )


def speak_button(text: str, key: str):
    safe_text = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
    st.components.v1.html(
        f"""
        <button id="speak-{key}" style="
            background: rgba(155,107,255,0.12); border: 1px solid {ACCENT}; color: #e9e4ff;
            border-radius: 999px; padding: 8px 18px; font-size: 0.85rem; font-weight: 600;
            cursor: pointer; white-space: nowrap; transition: background 0.15s;"
            onmouseover="this.style.background='rgba(155,107,255,0.24)'"
            onmouseout="this.style.background='rgba(155,107,255,0.12)'">
            🔊 Play question
        </button>
        <script>
        document.getElementById("speak-{key}").onclick = function() {{
            const u = new SpeechSynthesisUtterance("{safe_text}");
            u.rate = 0.95;
            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(u);
        }};
        </script>
        """,
        height=52,
    )


def score_gauge(value: int, title: str, color: str):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title, "font": {"size": 14, "color": "#cfcfe6"}},
            number={"font": {"size": 28, "color": "#f7f7fb"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#292942"},
                "bar": {"color": color},
                "bgcolor": "#12121f",
                "borderwidth": 1,
                "bordercolor": "#292942",
            },
        )
    )
    fig.update_layout(height=180, margin=dict(l=20, r=20, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def performance_radar(axes: dict, key: str):
    categories = list(axes.keys())
    values = list(axes.values())
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            line=dict(color=ACCENT_2),
            fillcolor="rgba(45,212,238,0.25)",
        )
    )
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 100], color="#8888a0", gridcolor="#292942"),
            angularaxis=dict(color="#cfcfe6", gridcolor="#292942"),
        ),
        showlegend=False,
        height=280,
        margin=dict(l=40, r=40, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=key)


def answer_radar_axes(answer) -> dict:
    ideal_pace_score = max(0.0, 1 - abs(answer.words_per_minute - 140) / 140) * 100
    filler_score = max(0.0, 1 - answer.filler_word_count / 5) * 100
    return {
        "Content": answer.content_score,
        "Delivery": answer.delivery_score,
        "Pace": round(ideal_pace_score),
        "Tone": round(answer.pitch_variation * 100),
        "Low fillers": round(filler_score),
    }


def star_checklist(star_components: dict):
    if not star_components or not any(star_components.values()):
        return
    cols = st.columns(4)
    for i, (key, label) in enumerate(STAR_LABELS.items()):
        present = star_components.get(key, False)
        with cols[i]:
            st.markdown(f"{'✅' if present else '⬜'} {label}")


def countdown_timer(seconds: int, key: str):
    st.components.v1.html(
        f"""
        <div style="display:flex; justify-content:center; width:100%;">
        <div id="timer-{key}" style="
            font-family: monospace; font-size: 1.05rem; font-weight: 600; color: #cfcfe6;
            text-align: center; background: rgba(255,255,255,0.04);
            border: 1px solid #292942; border-radius: 999px; padding: 6px 18px; display: inline-block;">
            ⏱ --:--
        </div>
        </div>
        <script>
        (function() {{
            let remaining = {seconds};
            const el = document.getElementById("timer-{key}");
            const tick = () => {{
                const m = String(Math.floor(remaining / 60)).padStart(2, '0');
                const s = String(remaining % 60).padStart(2, '0');
                el.innerHTML = "⏱ " + m + ":" + s;
                el.style.color = remaining <= 10 ? "#ff6b6b" : "#cfcfe6";
                if (remaining > 0) {{
                    remaining -= 1;
                    setTimeout(tick, 1000);
                }}
            }};
            tick();
        }})();
        </script>
        """,
        height=50,
    )


def level_up_answer(question: Question, key: str):
    """Independent-model rewrite of the candidate's own answer into a stronger version."""
    if not judge_client.is_configured():
        return
    answer = question.answer
    if not answer:
        return
    if answer.improved_answer:
        with st.expander("⚡ Level up this answer", expanded=False):
            st.markdown("**Your answer, leveled up by an independent model:**")
            st.success(answer.improved_answer)
        return
    if st.button("⚡ Level up this answer", key=f"levelup_{key}", help="Rewrite your answer into a stronger version"):
        with st.spinner("Rewriting your answer..."):
            try:
                answer.improved_answer = judge_client.rewrite_answer(
                    question.text, answer.transcript, st.session_state.session.role
                )
            except JudgeError as exc:
                st.warning(f"Couldn't reach the judge model for the rewrite: {exc}")
                return
        st.rerun()


def render_hiring_verdict(session: InterviewSession):
    """Independent hire/no-hire call from a second model (a true second opinion)."""
    if not judge_client.is_configured():
        with st.container(border=True):
            st.markdown("### 🔥 Independent hiring verdict")
            st.caption(
                "Powered by an independent second model — a second opinion independent "
                "beyond the Gemma 4 coach. Add a `JUDGE_API_KEY` (any OpenRouter key) to enable it."
            )
        return

    if session.hiring_verdict is None:
        with st.container(border=True):
            st.markdown("### 🔥 Independent hiring verdict")
            st.caption("An independent second model reviews the whole interview and makes a real call.")
            if st.button("Get the hiring manager's verdict", type="primary"):
                with st.spinner("The hiring manager is deliberating..."):
                    try:
                        session.hiring_verdict = judge_client.hiring_verdict(session)
                    except JudgeError as exc:
                        st.warning(f"Couldn't reach the judge model: {exc}")
                        return
                st.rerun()
        return

    v = session.hiring_verdict
    decision = v.get("decision", "Lean Hire")
    color = VERDICT_COLORS.get(decision, ACCENT)
    confidence = int(v.get("confidence", 0) or 0)
    with st.container(border=True):
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;">'
            f'<span style="background:{color};color:#0b0b12;font-weight:800;font-size:1.1rem;'
            f'border-radius:10px;padding:8px 18px;">{decision}</span>'
            f'<span style="color:#8888a0;">🔥 Independent judge · {confidence}% confidence</span>'
            f"</div>",
            unsafe_allow_html=True,
        )
        st.progress(confidence / 100)
        if v.get("rationale"):
            st.write(v["rationale"])
        vc = st.columns(2)
        with vc[0]:
            st.markdown(f"✅ **Case for:** {v.get('case_for', '—')}")
        with vc[1]:
            st.markdown(f"⚠️ **Biggest concern:** {v.get('case_against', '—')}")
        if v.get("standout_moment"):
            st.caption(f"💬 Standout: {v['standout_moment']}")


def jd_coverage(session: InterviewSession):
    """Which meaningful job-description keywords the candidate actually said out loud."""
    jd = (session.job_description or "").lower()
    words = re.findall(r"[a-zA-Z][a-zA-Z+#.]{2,}", jd)
    keywords, seen = [], set()
    for w in words:
        cleaned = w.strip(".")
        if cleaned in _STOPWORDS or cleaned in seen or len(cleaned) < 3:
            continue
        seen.add(cleaned)
        keywords.append(cleaned)
    keywords = keywords[:18]
    if not keywords:
        return None
    spoken = " ".join(q.answer.transcript.lower() for q in session.answered_questions)
    covered = [k for k in keywords if k in spoken]
    missing = [k for k in keywords if k not in spoken]
    pct = round(100 * len(covered) / len(keywords))
    return covered, missing, pct


def render_jd_coverage(session: InterviewSession):
    result = jd_coverage(session)
    if result is None:
        return
    covered, missing, pct = result
    with st.container(border=True):
        st.markdown(f"### 🎯 Job-description keyword coverage: {pct}%")
        st.caption("Key terms from the job description you actually worked into your answers.")
        st.progress(pct / 100)
        if covered:
            st.markdown("**Mentioned:** " + " ".join(f"`{k}`" for k in covered))
        if missing:
            st.markdown("**Not mentioned:** " + " ".join(f"`{k}`" for k in missing))


def setup_page():
    st.markdown('<h1 class="gradient-text">Pocket Interview Coach</h1>', unsafe_allow_html=True)
    st.write(
        "Practice out loud. Get coached on what you said **and** how you said it — "
        "pace, tone, filler words, and confidence."
    )
    hcols = st.columns(3)
    for col, (icon, title, body) in zip(
        hcols,
        [
            ("🗣️", "Speak", "Record real spoken answers to tailored questions."),
            ("🧠", "Get coached", "Gemma 4 scores your content **and** delivery."),
            ("🔥", "Get judged", "An independent model gives a hire / no-hire verdict."),
        ],
    ):
        with col:
            st.markdown(f"#### {icon} {title}")
            st.caption(body)

    prefill = st.session_state.drill_prefill or {}

    top_cols = st.columns([3, 1])
    with top_cols[1]:
        if st.button("🎲 Surprise me", use_container_width=True):
            st.session_state.surprise_pick = random.choice(SURPRISE_ROLES)
            st.rerun()

    surprise_role, surprise_company = st.session_state.surprise_pick or ("", "")

    # Timer controls live OUTSIDE the form: widgets inside an st.form don't rerun on change, so a
    # slider disabled by an in-form checkbox would never re-enable until submit. Out here, ticking
    # the checkbox reruns immediately and un-greys the slider.
    with st.container(border=True):
        tcol1, tcol2 = st.columns([1, 2])
        with tcol1:
            st.checkbox("⏱ Answer timer", key="timer_enabled")
        with tcol2:
            st.slider(
                "Timer duration (seconds)",
                30,
                300,
                step=15,
                key="timer_seconds",
                disabled=not st.session_state.timer_enabled,
            )

    with st.form("setup_form"):
        role = st.text_input(
            "Target role",
            value=surprise_role or prefill.get("role", ""),
            placeholder="e.g. Senior Backend Engineer",
        )
        st.caption("Presets: " + " · ".join(PRESET_ROLES))

        company = st.text_input(
            "Company (optional)", value=surprise_company or prefill.get("company", ""), placeholder="e.g. Acme Corp"
        )

        session_type = st.selectbox("Session type", list(SESSION_TYPES.keys()), format_func=lambda k: SESSION_TYPES[k]["label"])

        col1, col2 = st.columns(2)
        with col1:
            persona_key = st.selectbox(
                "Interviewer persona",
                [""] + list(PERSONAS.keys()),
                format_func=lambda k: "Default" if k == "" else k.replace("_", " ").title(),
            )
        with col2:
            panel_mode = st.checkbox("Panel mode (rotate personas)")

        job_description = st.text_area("Job description (optional)", height=100)

        num_questions = st.slider("Number of questions", 1, 10, 5)
        experience_level = st.select_slider("Experience level", options=EXPERIENCE_LEVELS, value=prefill.get("experience_level") or EXPERIENCE_LEVELS[1])

        resume_file = st.file_uploader("Resume (optional, PDF or text)", type=["pdf", "txt", "md"])

        sample_mode_label = st.radio("Sample answers", list(SAMPLE_ANSWER_MODES.keys()), horizontal=True, index=1)

        drill_focus = prefill.get("drill_focus", "")
        if drill_focus:
            st.info(f"🎯 Weakness drill: focused practice on **{drill_focus}**")

        submitted = st.form_submit_button("Start Mock Interview", use_container_width=True, type="primary")

    if submitted:
        if not role.strip():
            st.error("Enter a target role to generate your interview.")
            return

        st.session_state.sample_answer_mode = SAMPLE_ANSWER_MODES[sample_mode_label]
        st.session_state.surprise_pick = None

        resume_text = ""
        if resume_file is not None:
            suffix = os.path.splitext(resume_file.name)[1] or ".txt"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(resume_file.getvalue())
                tmp_path = tmp.name
            try:
                resume_text = resume_service.extract_text(tmp_path, resume_file.name)
            finally:
                os.unlink(tmp_path)

        with st.spinner("Generating your interview with Gemma 4..."):
            try:
                session = interview.create_session(
                    role=role.strip(),
                    job_description=job_description.strip(),
                    num_questions=num_questions,
                    company=company.strip(),
                    experience_level=experience_level,
                    resume_text=resume_text,
                    session_type=session_type,
                    persona=persona_key,
                    panel_mode=panel_mode,
                    drill_focus=drill_focus,
                    override=get_override(),
                )
            except Exception as exc:
                st.error(f"Couldn't start the session: {exc}")
                return

        st.session_state.session = session
        st.session_state.current_index = 0
        st.session_state.current_feedback = None
        st.session_state.drill_prefill = None
        st.session_state.page = "session"
        st.rerun()


def reset_answer(session: InterviewSession, question: Question):
    """Clear a recorded answer so the candidate can try the same question again."""
    question.answer = None  # the improved_answer rewrite lives on the Answer, so it's cleared too
    st.session_state.current_feedback = None
    # Bump the attempt counter so the mic_recorder gets a fresh key and won't replay old audio.
    st.session_state.attempts[question.id] = st.session_state.attempts.get(question.id, 0) + 1
    # Drop a not-yet-answered follow-up this answer had previously spawned, so re-recording
    # doesn't leave stale/duplicate follow-ups behind.
    i = session.questions.index(question)
    if i + 1 < len(session.questions):
        nxt = session.questions[i + 1]
        if nxt.is_dynamic and nxt.answer is None:
            session.questions.pop(i + 1)


def session_page():
    session: InterviewSession = st.session_state.session
    idx = st.session_state.current_index
    question: Question = session.questions[idx]

    top_cols = st.columns([2, 3, 1])
    with top_cols[0]:
        label = f"{session.role} @ {session.company}" if session.company else session.role
        st.markdown(f'<span class="persona-badge">{label}</span>', unsafe_allow_html=True)
    with top_cols[2]:
        st.markdown(f"**{idx + 1} / {len(session.questions)}**")
    st.progress((idx) / max(len(session.questions), 1))

    with st.container(border=True):
        header_cols = st.columns([3, 1])
        with header_cols[0]:
            tag_class = "qtag-followup" if question.is_dynamic else "qtag-question"
            tag_label = "🔍 Follow-up" if question.is_dynamic else "Question"
            badges = f'<span class="qtag {tag_class}">{tag_label}</span>'
            if question.persona:
                badges += f'&nbsp;<span class="persona-badge">{question.persona.replace("_", " ").title()}</span>'
            st.markdown(badges, unsafe_allow_html=True)
        with header_cols[1]:
            speak_button(question.text, key=question.id)
        st.markdown(f'<div class="question-text">{question.text}</div>', unsafe_allow_html=True)

        if question.answer is None:
            attempt = st.session_state.attempts.get(question.id, 0)
            with st.container(border=True):
                retry_note = " · this is a re-take" if attempt else ""
                st.markdown(
                    '<div style="text-align:center;">'
                    '<div style="font-size:2rem;">🎙️</div>'
                    '<div class="answer-hint" style="text-align:center;">'
                    "Take a breath, then answer out loud — speak naturally, as you would in the "
                    f"real interview.{retry_note}</div></div>",
                    unsafe_allow_html=True,
                )
                if st.session_state.timer_enabled:
                    countdown_timer(st.session_state.timer_seconds, key=f"timer_{question.id}_{attempt}")
                rec_cols = st.columns([1, 2, 1])
                with rec_cols[1]:
                    audio = mic_recorder(
                        start_prompt="🎙️  Record answer",
                        stop_prompt="⏹️  Stop & analyze",
                        just_once=True,
                        use_container_width=True,
                        format="wav",
                        key=f"rec_{question.id}_{attempt}",
                    )
                st.caption("Click **Record**, speak, then click **Stop & analyze**.")
            if audio and audio.get("bytes"):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(audio["bytes"])
                    tmp_path = tmp.name
                try:
                    with st.spinner("Transcribing and scoring your answer..."):
                        answer, follow_up = interview.process_answer(
                            session, question, tmp_path, get_override(), audio_bytes=audio["bytes"]
                        )
                    st.session_state.current_feedback = answer
                    if follow_up:
                        st.toast("Your interviewer added a follow-up question!", icon="🔍")
                except Exception as exc:
                    st.error(f"Couldn't analyze that answer: {exc}")
                finally:
                    os.unlink(tmp_path)
                st.rerun()
        else:
            answer = question.answer
            gcols = st.columns(2)
            with gcols[0]:
                score_gauge(answer.content_score, "Content", ACCENT)
            with gcols[1]:
                score_gauge(answer.delivery_score, "Delivery", ACCENT_2)

            mcols = st.columns(4)
            mcols[0].metric("Pace", f"{answer.words_per_minute:.0f} wpm")
            mcols[1].metric("Fillers", answer.filler_word_count)
            mcols[2].metric("Pauses", f"{answer.pause_ratio * 100:.0f}%")
            mcols[3].metric("Tone variation", f"{answer.pitch_variation * 100:.0f}%")

            if answer.audio_bytes:
                st.audio(answer.audio_bytes, format="audio/wav")

            with st.expander("Performance breakdown (radar)"):
                performance_radar(answer_radar_axes(answer), key=f"radar_{question.id}")

            star_checklist(answer.star_components)

            if len(answer.delivery_timeline) > 1:
                fig = go.Figure()
                ts = [p["t"] for p in answer.delivery_timeline]
                fig.add_trace(go.Scatter(x=ts, y=[p["words_per_minute"] for p in answer.delivery_timeline], name="Pace (wpm)", line=dict(color=ACCENT_2)))
                fig.add_trace(go.Scatter(x=ts, y=[p["pitch_variation"] * 100 for p in answer.delivery_timeline], name="Tone %", line=dict(color=ACCENT)))
                fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend=dict(font=dict(color="#cfcfe6")))
                fig.update_xaxes(color="#8888a0")
                fig.update_yaxes(color="#8888a0")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            st.info(f"**Content feedback:** {answer.content_feedback}")
            st.info(f"**Voice & tone feedback:** {answer.delivery_feedback}")

            if st.session_state.sample_answer_mode == "after" and question.sample_answer:
                with st.expander("Sample answer"):
                    st.write(question.sample_answer)

            level_up_answer(question, key=f"session_{question.id}")

            is_last = idx + 1 == len(session.questions)
            nav_cols = st.columns([1, 1])
            with nav_cols[0]:
                if st.button("🔄 Re-record", key=f"rerec_{question.id}", use_container_width=True):
                    reset_answer(session, question)
                    st.rerun()
            with nav_cols[1]:
                if st.button("View Report" if is_last else "Next Question →", type="primary", use_container_width=True):
                    st.session_state.current_feedback = None
                    if is_last:
                        st.session_state.page = "report"
                    else:
                        st.session_state.current_index += 1
                    st.rerun()


def compute_weakest_area(session: InterviewSession) -> str:
    answered = session.answered_questions
    n = len(answered)
    scores = {
        "filler words": max(0.0, 1 - sum(a.answer.filler_word_count for a in answered) / n / 5),
        "pausing and hesitation": max(0.0, 1 - sum(a.answer.pause_ratio for a in answered) / n / 0.3),
        "vocal tone and expressiveness": sum(a.answer.pitch_variation for a in answered) / n,
        "voice steadiness": sum(a.answer.volume_consistency for a in answered) / n,
        "answer structure and specificity": sum(a.answer.content_score for a in answered) / n / 100,
    }
    return min(scores, key=scores.get)


def report_page():
    session: InterviewSession = st.session_state.session

    if not session.summary:
        with st.spinner("Building your report..."):
            try:
                interview.build_report(session, get_override())
            except Exception as exc:
                st.error(f"Couldn't build the report: {exc}")
                return

    st.markdown('<h1 class="gradient-text">Session Report</h1>', unsafe_allow_html=True)

    readiness = session.readiness
    celebrate_key = f"celebrated_{session.id}"
    if readiness["grade"] in ("A", "B") and not st.session_state.get(celebrate_key):
        st.balloons()
        st.session_state[celebrate_key] = True

    st.markdown(
        f"### Interview Readiness: {readiness['score']}/100 &nbsp; "
        f'<span class="persona-badge">Grade {readiness["grade"]}</span>',
        unsafe_allow_html=True,
    )

    gcols = st.columns(2)
    with gcols[0]:
        score_gauge(session.avg_content_score, "Avg Content", ACCENT)
    with gcols[1]:
        score_gauge(session.avg_delivery_score, "Avg Delivery", ACCENT_2)

    st.write(session.summary)

    render_hiring_verdict(session)
    render_jd_coverage(session)

    answered = session.answered_questions

    if len(answered) >= 1:
        avg_axes = {}
        for key in ["Content", "Delivery", "Pace", "Tone", "Low fillers"]:
            avg_axes[key] = round(sum(answer_radar_axes(q.answer)[key] for q in answered) / len(answered))
        with st.expander("Overall performance breakdown (radar)"):
            performance_radar(avg_axes, key="report_radar")

    export_cols = st.columns(3)
    with export_cols[0]:
        st.download_button(
            "📄 Download PDF",
            data=export_service.generate_report_pdf(session),
            file_name="interview_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    with export_cols[1]:
        st.download_button(
            "📝 Download transcript",
            data=export_service.generate_transcript_text(session),
            file_name="interview_transcript.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with export_cols[2]:
        st.download_button(
            "🗂 Download JSON",
            data=export_service.generate_report_json(session),
            file_name="interview_report.json",
            mime="application/json",
            use_container_width=True,
        )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[f"Q{i+1}" for i in range(len(answered))], y=[q.answer.content_score for q in answered], name="Content", line=dict(color=ACCENT)))
    fig.add_trace(go.Scatter(x=[f"Q{i+1}" for i in range(len(answered))], y=[q.answer.delivery_score for q in answered], name="Delivery", line=dict(color=ACCENT_2)))
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend=dict(font=dict(color="#cfcfe6")), yaxis=dict(range=[0, 100]))
    fig.update_xaxes(color="#8888a0")
    fig.update_yaxes(color="#8888a0")
    st.subheader("Score trend")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.subheader("Top actions")
    for action in session.top_actions:
        st.markdown(f"- {action}")

    if len(answered) >= 2:
        weakest = compute_weakest_area(session)
        with st.container(border=True):
            st.markdown(f"🎯 **Your biggest opportunity:** {weakest}")
            if st.button("Drill this weak area"):
                st.session_state.drill_prefill = {
                    "role": session.role,
                    "company": session.company,
                    "experience_level": session.experience_level,
                    "drill_focus": weakest,
                }
                st.session_state.page = "setup"
                st.session_state.session = None
                st.rerun()

    with st.container(border=True):
        st.subheader("Personalized cheat sheet")
        if not session.cheat_sheet:
            if st.button("✨ Generate my cheat sheet"):
                with st.spinner("Generating..."):
                    try:
                        interview.build_cheat_sheet(session, get_override())
                    except Exception as exc:
                        st.error(f"Couldn't generate a cheat sheet: {exc}")
                st.rerun()
        else:
            st.markdown(session.cheat_sheet)

    st.subheader("Per-question detail")
    for i, q in enumerate(answered):
        with st.expander(f"Q{i + 1}: {q.text}"):
            st.caption(f"{q.answer.content_score}/100 content · {q.answer.delivery_score}/100 delivery")
            st.write(f"*\"{q.answer.transcript}\"*")
            st.write(q.answer.content_feedback)
            st.write(q.answer.delivery_feedback)
            if st.session_state.sample_answer_mode == "end" and q.sample_answer:
                st.markdown("**Sample answer:**")
                st.write(q.sample_answer)
            level_up_answer(q, key=f"report_{q.id}")

    if st.button("Start New Interview", use_container_width=True):
        st.session_state.page = "setup"
        st.session_state.session = None
        st.rerun()


init_state()
render_sidebar()

if st.session_state.page == "setup":
    setup_page()
elif st.session_state.page == "session":
    session_page()
elif st.session_state.page == "report":
    report_page()
