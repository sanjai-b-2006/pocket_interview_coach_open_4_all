# Pocket Interview Coach — Open Edition

A voice-first AI mock-interview coach that scores **what you say** *and* **how you say it** — pace,
tone, filler words, pauses, and confidence — not just the transcript.

This is the **open, run-anywhere edition**: everything runs through **OpenRouter** with a single API
key, so you can fork it, drop in your own key, and deploy your own instance in minutes. No
provider-specific setup.

> Two AIs, two roles: a **coach** model (Gemma 4 by default) generates questions and scores your
> content and delivery, while a separate **independent "hiring manager"** model reviews the whole
> interview and makes a real hire / no-hire call. Two different models means a genuine second
> opinion, not one model grading itself.

## Features

- Pick a role (plus optional company, job description, resume, persona, panel mode, difficulty)
- Record spoken answers in the browser; local speech-to-text + delivery-metric analysis
- **Content** and **delivery** scoring (0–100) with plain-English coaching for both
- STAR-method checklist, performance radar, per-answer delivery timeline
- AI **devil's-advocate follow-up questions** inserted mid-interview
- **Independent hiring verdict** — a second model gives a Strong-Hire → No-Hire call with confidence
- **Level up this answer** — rewrites your own answer into a stronger version (grounded, no invented facts)
- Readiness grade, personalized cheat sheet, JD keyword coverage
- **Re-record** any answer, text-to-speech question playback
- Export report as PDF / JSON / transcript
- **Bring-your-own-key** — override the API key/model right in the sidebar

## How it works

Your voice → `faster-whisper` (speech-to-text) + `librosa` (prosody: pace, pitch, fillers, pauses,
volume) → **coach model** (scores content + delivery, asks follow-ups, writes the cheat sheet) →
**independent judge model** (hiring verdict + answer rewrite) → report + exports.

## Running locally

```bash
python -m venv venv
source venv/Scripts/activate        # or venv/bin/activate on macOS/Linux
pip install -r requirements.txt

cp .env.example .env
# edit .env: set OPENROUTER_API_KEY

streamlit run app.py
```

Open http://localhost:8501.

## Running with Docker

```bash
docker build -t pocket-interview-open .
docker run -p 8501:8501 --env-file .env pocket-interview-open
```

## Two ways to run it publicly

- **Everyone brings their own key (recommended for a public link).** Do **not** set
  `OPENROUTER_API_KEY`. On load, each visitor sees a one-time screen asking for their own OpenRouter
  key — it's used for both the coach and the judge, kept in their browser session only, and never
  stored. Your credits are never touched.
- **You provide the key for everyone.** Set `OPENROUTER_API_KEY` in Secrets and the app just works
  with no gate — but every visitor spends *your* credits. (Set `REQUIRE_USER_KEY = "true"` to force
  the bring-your-own-key screen even when a default key exists.)

## Deploying to Streamlit Community Cloud (free)

1. Push this repo to GitHub (public).
2. Go to **share.streamlit.io** → **New app** → select this repo, branch `main`, main file `app.py`.
3. Under **Advanced settings → Secrets**, either leave it empty (visitors bring their own key) or add:
   ```toml
   OPENROUTER_API_KEY = "your_openrouter_key"
   COACH_MODEL = "google/gemma-4-26b-a4b-it"
   JUDGE_MODEL = "openai/gpt-4o-mini"
   ```
4. Deploy. (Set the app's Python version to **3.11** in Settings for prebuilt wheels.)

## Configuration

| Variable | Purpose |
|---|---|
| `OPENROUTER_API_KEY` | Your OpenRouter key — powers both coach and judge |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` (default) |
| `COACH_MODEL` | Coach model slug (default `google/gemma-4-26b-a4b-it`) |
| `JUDGE_MODEL` | Independent judge model (default `openai/gpt-4o-mini`) |
| `JUDGE_API_KEY` / `JUDGE_BASE_URL` | Optional — put the judge on a different key/endpoint |
| `REQUIRE_USER_KEY` | `true` to always show the bring-your-own-key screen (default `false`) |
| `ASR_MODEL_SIZE` | `base` (default) — use `tiny` on memory-constrained hosts |
| `ENABLE_PITCH_ANALYSIS` | `true` (default) — `false` to skip the heavier pitch pass |

Any OpenAI-compatible endpoint works — swap `OPENROUTER_BASE_URL`/model slugs to point at OpenAI,
Together, a local server, etc. Pick any two models you like for coach vs. judge.

## Project structure

```
app.py                   Single-file Streamlit UI (setup / session / report)
services/
  config.py               Env-var settings (OpenRouter)
  llm.py                  Coach model: questions, scoring, follow-ups, cheat sheet
  judge.py                Independent judge model: hiring verdict + answer rewrite
  asr.py                  faster-whisper speech-to-text
  prosody.py              librosa delivery metrics (pace, pitch, pauses, fillers, timeline)
  resume.py               PDF/text resume extraction
  models.py               In-memory session/question/answer dataclasses (no database)
  interview.py            Orchestrates create_session / process_answer / build_report / cheat_sheet
```

No database — each browser session holds its own state in Streamlit's `session_state`.
