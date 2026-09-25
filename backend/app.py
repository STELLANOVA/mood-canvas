"""Mood Canvas API. Run: uvicorn app:app --reload --port 8000"""
import asyncio
import datetime as dt
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import agent
import services as S
from prompts import exhibition_prompt

app = FastAPI(title="Mood Canvas")
STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")

DOW = {1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 5: "Friday", 6: "Saturday", 7: "Sunday"}


class EntryIn(BaseModel):
    text: str = Field(min_length=3, max_length=4000)
    did_experiment: Optional[bool] = None


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.post("/api/entries")
async def create_entry(body: EntryIn):
    keyword_risk = S.crisis_check(body.text)
    try:
        signals = await S.extract_signals(body.text)  # raw text goes ONLY to Liquid
    except Exception as e:  # noqa: BLE001
        if keyword_risk:
            agent.pause_for_safety()
            return {"entry": None, "crisis": S.CRISIS_RESOURCES, "resources": []}
        raise HTTPException(502, f"The mood model is unreachable: {e}")

    risk = keyword_risk or signals["risk_flag"]
    entry_id = uuid.uuid4().hex[:12]
    active = agent.load()["active"]
    topic = signals["stressors"][0] if signals["stressors"] else "general"

    if risk:
        agent.pause_for_safety()
        image_path, resources = "", []
    else:
        image_path, resources = await asyncio.gather(
            S.paint(signals, entry_id), S.find_resources(topic))

    row = {
        "entry_id": entry_id, "user_id": S.USER_ID,
        "ts": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        **{k: signals[k] for k in ("mood", "energy", "anxiety", "sleep_hours", "emotions",
                                   "stressors", "positives", "palette", "title")},
        "risk_flag": int(risk), "image_path": image_path,
        "exp_id": (active or {}).get("driver", ""), "exp_done": int(bool(active and body.did_experiment)),
    }
    try:
        await S.tb_ingest([row])
    except Exception as e:  # noqa: BLE001
        print(f"[tinybird] {e}")

    return {
        "entry": {**row, "reflection": signals["reflection"]},
        "crisis": S.CRISIS_RESOURCES if risk else None,
        "resources": resources,
    }


# ---------- voice check-ins: audio -> local Liquid ASR -> the same text path ----------
MAX_AUDIO_BYTES = 10 * 1024 * 1024  # ~5 minutes of 16 kHz mono WAV


@app.get("/api/voice")
async def voice_status():
    return {"available": await S.audio_available()}


@app.post("/api/voice")
async def voice_entry(request: Request, did_experiment: Optional[bool] = None):
    wav = await request.body()  # held in memory only; never written to disk or logged
    if not wav.startswith(b"RIFF") or len(wav) > MAX_AUDIO_BYTES:
        raise HTTPException(400, "Expected a WAV recording under 10 MB.")
    try:
        text = await S.transcribe(wav)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"The voice model is unreachable: {e}")
    del wav
    if len(text) < 3:
        raise HTTPException(422, "Couldn't make out any words. Try again a little closer to the mic.")
    result = await create_entry(EntryIn(text=text[:4000], did_experiment=did_experiment))
    return {**result, "transcript": text}


# ---------- weekly exhibitions: a week's paintings composed into one panorama ----------
_exhibit_lock = asyncio.Lock()


def exhibition_weeks() -> list:
    """7-day windows aligned with the agent's cycles, labeled from the lab notebook."""
    card, rows = agent.load(), [r for r in S.mirror_rows() if r["user_id"] == S.USER_ID]
    if not rows:
        return []
    labels = {e["sim_week_of"]: e["title"].split(": ", 1)[-1] for e in agent.read_log()
              if e["phase"] == "act" and e["title"].startswith("Started")}
    start0 = dt.date.fromisoformat(card["baseline"]["start"])
    weeks = {}
    for r in sorted(rows, key=lambda r: r["ts"]):
        k = (dt.date.fromisoformat(r["ts"][:10]) - start0).days // 7
        weeks.setdefault(k, []).append(r)
    out = []
    for k, g in sorted(weeks.items()):
        start = start0 + dt.timedelta(days=7 * k)
        key = start.isoformat()
        existing = sorted(S.EXHIBIT_DIR.glob(f"week-{key}.*"))
        out.append({
            "start": key, "end": (start + dt.timedelta(days=6)).isoformat(),
            "label": labels.get(key) or ("Baseline" if 0 <= k < 2 else "Live check-ins"),
            "experiment": key in labels,
            "entries": len(g), "avg_mood": round(sum(r["mood"] for r in g) / len(g), 1),
            "moods": [r["mood"] for r in g],
            "paintings": [r["image_path"] for r in g if r.get("image_path")],
            "image": f"/static/exhibitions/{existing[0].name}" if existing else "",
        })
    return out


@app.get("/api/exhibitions")
async def exhibitions():
    return exhibition_weeks()


@app.post("/api/exhibitions/{start}")
async def create_exhibition(start: str):
    week = next((w for w in exhibition_weeks() if w["start"] == start), None)
    if not week:
        raise HTTPException(404, "No check-ins that week.")
    if week["image"]:
        return week
    if not week["paintings"]:
        raise HTTPException(422, "This week has no paintings to exhibit yet.")
    if _exhibit_lock.locked():
        raise HTTPException(409, "Another exhibition is being composed. Try again in a moment.")
    async with _exhibit_lock:
        paths = [STATIC / p.removeprefix("/static/") for p in week["paintings"]]
        image = await S.compose_exhibition([p for p in paths if p.exists()],
                                           exhibition_prompt(week["moods"]), f"week-{start}")
    if not image:
        raise HTTPException(502, "The painter couldn't compose this week. Try again.")
    return {**week, "image": image}


# ---------- the agent ----------
_agent_lock = asyncio.Lock()


def agent_view(card: dict, message: str = "") -> dict:
    return {"card": card, "message": message, "log": agent.read_log(),
            "card_tokens": agent.card_tokens(card), "entries_seen": len(S.mirror_rows()),
            "drivers": agent.DRIVERS}


@app.get("/api/agent")
async def agent_state():
    return agent_view(agent.load())


@app.post("/api/agent/step")
async def agent_step(simulate: bool = True, resume: bool = False):
    if _agent_lock.locked():
        raise HTTPException(409, "The agent is already running a cycle.")
    async with _agent_lock:
        r = await agent.step(simulate=simulate, resume=resume)
    return agent_view(r["card"], r["message"])


@app.get("/api/gallery")
async def gallery(limit: int = 14):
    try:
        return await S.tb_pipe("gallery", user_id=S.USER_ID, limit=limit)
    except Exception as e:  # noqa: BLE001
        print(f"[tinybird] gallery, using local mirror: {e}")
        return sorted(S.mirror_rows(), key=lambda r: r["ts"], reverse=True)[:limit]


@app.get("/api/resources")
async def resources(topic: str = "general"):
    return await S.find_resources(topic)


@app.get("/api/insights")
async def insights(days: int = 45):
    try:
        daily, stressors, weekday = await asyncio.gather(
            S.tb_pipe("daily_mood", user_id=S.USER_ID, days=days),
            S.tb_pipe("stressor_impact", user_id=S.USER_ID, days=days),
            S.tb_pipe("weekday_pattern", user_id=S.USER_ID),
        )
    except Exception as e:  # noqa: BLE001
        print(f"[tinybird] insights, using local mirror: {e}")
        daily, stressors, weekday = local_insights(days)
    return {"daily": daily, "stressors": stressors, "weekday": weekday,
            "sentences": build_sentences(daily, stressors, weekday)}


def local_insights(days: int):
    """Same shapes as the Tinybird endpoints, computed from the signals-only mirror."""
    since = (dt.datetime.utcnow() - dt.timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    rows = [r for r in S.mirror_rows() if r["user_id"] == S.USER_ID and r["ts"] > since]
    avg = lambda xs: round(sum(xs) / len(xs), 1) if xs else None  # noqa: E731
    by_day, by_dow, by_s = {}, {}, {}
    for r in rows:
        by_day.setdefault(r["ts"][:10], []).append(r)
        by_dow.setdefault(dt.date.fromisoformat(r["ts"][:10]).isoweekday(), []).append(r)
        for st in r["stressors"]:
            by_s.setdefault(st, []).append(r)
    daily = [{"day": d, "mood": avg([r["mood"] for r in g]), "energy": avg([r["energy"] for r in g]),
              "anxiety": avg([r["anxiety"] for r in g]),
              "sleep": avg([r["sleep_hours"] for r in g if r.get("sleep_hours") is not None])}
             for d, g in sorted(by_day.items())]
    stressors = sorted([{"stressor": k, "entries": len(g), "avg_mood": avg([r["mood"] for r in g]),
                         "avg_anxiety": avg([r["anxiety"] for r in g])} for k, g in by_s.items()],
                       key=lambda x: -x["entries"])
    weekday = [{"dow": k, "mood": avg([r["mood"] for r in g]), "anxiety": avg([r["anxiety"] for r in g]),
                "entries": len(g)} for k, g in sorted(by_dow.items())]
    return daily, stressors, weekday


def build_sentences(daily, stressors, weekday) -> list[str]:
    out = []
    if len(weekday) >= 3:
        tense = max(weekday, key=lambda w: w["anxiety"])
        bright = max(weekday, key=lambda w: w["mood"])
        out.append(f"Anxiety runs highest on {DOW[tense['dow']]}s, averaging {tense['anxiety']} out of 10.")
        if bright["dow"] != tense["dow"]:
            out.append(f"{DOW[bright['dow']]}s tend to be your brightest days.")

    slept = [d for d in daily if d.get("sleep") is not None]
    short = [d["mood"] for d in slept if d["sleep"] < 6]
    rested = [d["mood"] for d in slept if d["sleep"] >= 7]
    if short and rested:
        a, b = sum(short) / len(short), sum(rested) / len(rested)
        if abs(b - a) >= 0.5:
            out.append(f"On days with under 6 hours of sleep your mood averages {a:.1f}, "
                       f"compared with {b:.1f} after 7 or more hours.")

    if stressors and daily:
        top = stressors[0]
        overall = sum(d["mood"] for d in daily) / len(daily)
        out.append(f"{top['stressor'].capitalize()} came up in {top['entries']} check-ins. "
                   f"Mood on those days averages {top['avg_mood']}, versus {overall:.1f} overall.")
    return out
