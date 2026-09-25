# Mood Canvas: context for Claude Code

## What this is
A 5-hour solo build for the **Long Horizon Agents** hackathon ("build agents that preserve what matters": plan, act, observe, self-correct across a full cycle without drowning in their own history; 3+ event tech partners). Event partners: Codex, OpenAI, Liquid AI, Nimble, Tinybird, Black Forest Labs. We use four: Liquid AI, Nimble, Tinybird, Black Forest Labs. AWS is not an event partner. The app runs locally.

Mood Canvas is a wellbeing agent that runs weekly n-of-1 experiments. Daily check-ins become paintings and signals. Each week the agent observes (Tinybird), corrects (rule-based verdict vs a baseline), plans the next hypothesis (planner LLM over aggregates), and acts (starts a small habit experiment, cites Nimble sources). Its only context between cycles is a memory card of about 360 tokens, which is both the long-horizon memory story and the privacy story.

## Architecture
- `backend/app.py` is the FastAPI app. `POST /api/entries` runs the crisis keyword check, then Liquid extraction, then FLUX painting and Nimble resources in parallel, then Tinybird ingest. Also `GET /api/gallery`, `/api/insights`, `/api/resources`.
- `backend/services.py` holds all partner integrations. Each fails soft so the demo never crashes.
  - Liquid: OpenAI-compatible `/chat/completions` (`LIQUID_BASE_URL`, default local llama-server on :8080).
  - BFL: `POST https://api.bfl.ai/v1/{BFL_MODEL}` with `x-key` header, poll `polling_url` until `Ready`, download `result.sample` immediately because the URLs expire.
  - Nimble: `POST https://sdk.nimbleway.com/v2/search` with Bearer auth, `include_domains` limited to trusted health sites.
  - Tinybird: Events API `POST /v0/events?name=mood_entries` (NDJSON), endpoints via `GET /v0/pipes/{name}.json`.
- `backend/agent.py` is the agent. `step()` runs one cycle: observe_and_correct, then plan_and_act. The memory card is `backend/data/agent_state.json`. The lab notebook is `backend/data/agent_log.jsonl`, also sent to Tinybird `agent_events`, and is never fed back to the agent. Verdicts are rules (`verdict()`); the LLM only picks from a closed driver list and writes prose, with fallbacks. `GET /api/agent`, `POST /api/agent/step?simulate=true&resume=false`.
- Demo mode: each step simulates a week via `seed.synth_day()`. Planted truth: short sleep drives mood; work stress is a confound, and sleep is under-reported (~40%). The rule-based planner tests work, then exercise, then sleep (supported). An LLM planner may choose a different order.
- `backend/data/entries.jsonl` is a signals-only local mirror of everything ingested. The agent, gallery and insights fall back to it when Tinybird is unreachable.
- Voice: `POST /api/voice` (raw WAV body) -> `services.transcribe()` via Liquid's `llama-liquid-audio-server` on :8082 (`LIQUID_AUDIO_URL`; streaming-only API, no /health route) -> the same `create_entry` path. Audio is held in memory only. The browser converts MediaRecorder output to 16 kHz WAV (`toWav16k`). The mic button is hidden when the audio server is down.
- `services.scrub_art_fields()` replaces the title and scene with a stock landscape if they reuse entry words or indoor nouns, because the title goes to Tinybird and the scene to FLUX.
- Exhibitions: `GET /api/exhibitions` lists 7-day windows aligned with the agent's baseline start, labeled from lab-notebook "act" entries (read from the local mirror). `POST /api/exhibitions/{start}` sends up to 8 paintings as base64 `input_image`, `input_image_2`… to flux-2-pro (FLUX.1 Kontext is single-image) with `prompts.exhibition_prompt(moods)`. Saved as `static/exhibitions/week-{start}.jpg`. Simulated weeks paint 3 days spread across the week (`simulate_days(spread=True)`), in parallel.
- Painting styles: `prompts.STYLES` (id -> label, public-domain painter credit, technique-only prompt). `art_prompt(signals, style)`; "mood" keeps the original mood-driven look. `EntryIn.style` / `/api/voice?style=`, `GET /api/styles`; the UI remembers the choice in localStorage. `services.trim_signature_margin()` crops every FLUX image (bottom 10%, sides 5%) with Pillow to remove fake signatures. Simulated weeks always use "mood".
- Sharing is client-side only: `drawCard()` renders painting + title + chosen caption (+ optional date) to a canvas. No entry text, scores or tags.
- `backend/prompts.py` has the Liquid extraction prompt (JSON-only, closed vocabularies for stressors and positives) and the FLUX art prompt builder.
- `backend/seed.py` clears Tinybird (through the `tb` CLI login) and the local mirror, seeds a 14-day baseline six weeks back, and resets the agent. `--paint N` paints the last N baseline days. `synth_day()` also simulates experiment weeks for the agent.
- `backend/static/index.html` is a single-file frontend. Its headline feature is the gallery wall, where each painting hangs higher on better days (margin-bottom = (mood-1)*24px). Chart.js draws the trend chart.
- `tinybird/` holds the `mood_entries` (now with `exp_id`, `exp_done`) and `agent_events` datasources, plus 6 endpoint pipes: `daily_mood`, `stressor_impact`, `weekday_pattern`, `gallery`, and the agent's `window_stats` and `driver_scan`.

## Non-negotiable rules
- **Privacy:** Raw journal text goes ONLY to the Liquid model. Never log it, store it, or send it to FLUX, Nimble, or Tinybird. FLUX gets the abstract scene and palette; Nimble gets a topic keyword.
- **Safety:** If crisis language is detected (keywords or the model's `risk_flag`), skip painting and resources, show the 988 / Crisis Text Line card, and pause the agent until the user resumes. Never let the model diagnose or give medical advice. Experiments are limited to the four low-risk habits in `agent.DRIVERS`.
- **Demo reliability over features:** Every external call must degrade gracefully.

## Status and notes
- Verified against the real services: Tinybird (pipes match the local fallback exactly), Nimble search, BFL flux-2-pro (~9 s per painting, ~20 s per weekly exhibition), local Liquid LFM2-2.6B (~2.5 s per extraction on Apple Silicon) and LFM2.5-Audio (~0.4 s per transcription). A live check-in takes ~19 s end to end, mostly FLUX.
- Tinybird Forward, deployed with `tb --cloud deploy` from `tinybird/`. The app uses the scoped `mood_canvas_app` token (append to both datasources, read the pipes), defined in the datafiles. It can't truncate tables, so `seed.py` uses the `tb` CLI login and aborts rather than seed on top of old data.
- Tinybird gotchas: a node can't share its pipe's name, and a CTE name in WITH is read as a missing table (use a separate node instead).
- Liquid lessons: extraction uses `response_format` json_schema (constrained decoding, 0/18 failures, falls back to prompt-only on 400/422). Entries under 6 words are grounded by keyword (`ground_short_entry`) because the model invents stressors for "ok". As planner, the 2.6B model misranked candidates and garbled numbers, so the ranking decides, the LLM only breaks near-ties (>= 75% of top score) and writes the "why", and lessons and notes are rule text.
- Screenshots and diagrams: `docs/screenshots/`.

## Commands
```bash
cd backend && pip install -r requirements.txt
python seed.py --paint 3   # clears Tinybird, 14 baseline days, resets the agent
uvicorn app:app --reload --port 8000
```
