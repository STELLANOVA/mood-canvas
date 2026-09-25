"""All partner integrations live here. Each one fails soft so the demo never hard-crashes."""
import asyncio
import base64
import json
import os
import re
from pathlib import Path

import httpx
from dotenv import load_dotenv

from prompts import EXTRACT_SCHEMA, EXTRACT_SYSTEM, POSITIVES, STRESSORS, art_prompt

load_dotenv()

# Liquid AI: any OpenAI-compatible endpoint (local llama-server by default, or a hosted one such as OpenRouter)
LIQUID_BASE_URL = os.getenv("LIQUID_BASE_URL", "http://localhost:8080/v1").rstrip("/")
LIQUID_API_KEY = os.getenv("LIQUID_API_KEY", "not-needed")
LIQUID_MODEL = os.getenv("LIQUID_MODEL", "LFM2-2.6B")

# Black Forest Labs
BFL_API_KEY = os.getenv("BFL_API_KEY", "")
BFL_BASE = os.getenv("BFL_BASE", "https://api.bfl.ai/v1").rstrip("/")
BFL_MODEL = os.getenv("BFL_MODEL", "flux-2-pro")

# Nimble
NIMBLE_API_KEY = os.getenv("NIMBLE_API_KEY", "")
NIMBLE_BASE = os.getenv("NIMBLE_BASE", "https://sdk.nimbleway.com").rstrip("/")

# Tinybird
TB_HOST = os.getenv("TB_HOST", "https://api.tinybird.co").rstrip("/")
TB_TOKEN = os.getenv("TB_TOKEN", "")

# Liquid audio: LFM2.5-Audio via Liquid's llama-liquid-audio-server (streaming-only API). Audio is
# transcribed locally and never stored; the transcript then follows the normal text path.
LIQUID_AUDIO_URL = os.getenv("LIQUID_AUDIO_URL", "http://127.0.0.1:8082/v1").rstrip("/")

# Agent planner: any OpenAI-compatible endpoint. It only ever sees aggregates, never journal text,
# so it can be a larger model than the private extractor. Defaults to the Liquid endpoint.
PLANNER_BASE_URL = os.getenv("PLANNER_BASE_URL", LIQUID_BASE_URL).rstrip("/")
PLANNER_API_KEY = os.getenv("PLANNER_API_KEY", LIQUID_API_KEY)
PLANNER_MODEL = os.getenv("PLANNER_MODEL", LIQUID_MODEL)

USER_ID = os.getenv("DEMO_USER_ID", "demo")
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
MIRROR = DATA_DIR / "entries.jsonl"  # signals-only local mirror, used when Tinybird is unreachable
PAINT_DIR = Path(__file__).parent / "static" / "paintings"
PAINT_DIR.mkdir(parents=True, exist_ok=True)
EXHIBIT_DIR = Path(__file__).parent / "static" / "exhibitions"
EXHIBIT_DIR.mkdir(parents=True, exist_ok=True)


# ---------- Safety ----------
CRISIS_PATTERNS = [
    r"\bkill (my ?self|myself)\b", r"\bsuicid", r"\bend (it all|my life)\b",
    r"\bwant(ed)? to die\b", r"\bdon'?t want to (live|be alive|be here)\b",
    r"\bself[- ]?harm", r"\bhurt(ing)? myself\b", r"\bcut(ting)? myself\b",
    r"\bno reason to live\b", r"\bbetter off without me\b",
]

CRISIS_RESOURCES = {
    "message": "It sounds like you might be going through something really painful. "
               "You don't have to hold this alone. Talking to someone right now can help.",
    "options": [
        {"name": "988 Suicide & Crisis Lifeline (US)", "action": "Call or text 988", "url": "https://988lifeline.org"},
        {"name": "Crisis Text Line (US)", "action": "Text HOME to 741741", "url": "https://www.crisistextline.org"},
        {"name": "Outside the US", "action": "Find a local helpline", "url": "https://findahelpline.com"},
    ],
}


def crisis_check(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in CRISIS_PATTERNS)


# ---------- Liquid AI: private signal extraction ----------
def _parse_json(raw: str) -> dict:
    raw = raw.replace("```json", "").replace("```", "").strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object in model output")
    return json.loads(raw[start:end + 1])


def _clamp(v, lo=1, hi=10, default=5) -> int:
    try:
        return max(lo, min(hi, int(round(float(v)))))
    except (TypeError, ValueError):
        return default


def _default_palette(mood: int) -> list[str]:
    if mood <= 3:
        return ["#3E4A61", "#6D7A8F", "#B8A89A"]
    if mood <= 6:
        return ["#6F8FAF", "#C9B79C", "#E8DCC8"]
    return ["#F2B880", "#8FC1A9", "#F7E3AF"]


def normalize(d: dict) -> dict:
    mood = _clamp(d.get("mood"))
    sleep = d.get("sleep_hours")
    try:
        sleep = round(float(sleep), 1) if sleep is not None else None
        if sleep is not None and not 0 <= sleep <= 16:
            sleep = None
    except (TypeError, ValueError):
        sleep = None
    palette = [c for c in d.get("palette", []) if isinstance(c, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", c)]
    stressors = [s for s in d.get("stressors", []) if s in STRESSORS and s != "none"][:3]
    if sleep is not None and sleep < 6 and "sleep" not in stressors:  # the agent relies on this tag
        stressors = (["sleep"] + stressors)[:3]
    positives = [p for p in d.get("positives", []) if p in POSITIVES and p != "none"][:3]
    return {
        "mood": mood,
        "energy": _clamp(d.get("energy")),
        "anxiety": _clamp(d.get("anxiety")),
        "sleep_hours": sleep,
        "emotions": [str(e).lower()[:20] for e in d.get("emotions", [])][:3],
        "stressors": stressors,
        "positives": positives,
        "risk_flag": bool(d.get("risk_flag", False)),
        "title": str(d.get("title") or "Untitled day")[:60],
        "palette": palette[:3] if len(palette) >= 3 else _default_palette(mood),
        "scene": str(d.get("scene") or "Soft layered horizon under shifting light.")[:240],
        "reflection": str(d.get("reflection") or "Thank you for checking in with yourself today.")[:300],
    }


# Grounding for very short entries: small models fill silence with plausible stressors ("ok" -> work),
# which would skew the agent's driver scan. Keep a stressor only if the entry actually hints at it.
SHORT_ENTRY_WORDS = 6
STRESSOR_HINTS = {
    "work": r"work|job|boss|manager|meeting|deadline|office|shift|project|client",
    "school": r"school|class|exam|homework|study|uni|college",
    "relationships": r"partner|boyfriend|girlfriend|husband|wife|date|breakup|relationship",
    "family": r"mom|mum|dad|parent|sister|brother|family|kid|child",
    "health": r"sick|ill|pain|headache|doctor|health",
    "money": r"money|rent|bill|debt|broke|pay",
    "sleep": r"sleep|slept|tired|exhausted|insomnia|awake|nap",
    "loneliness": r"alone|lonely|isolated",
    "news": r"news|politic|war|election",
}


def ground_short_entry(text: str, signals: dict) -> dict:
    if len(text.split()) >= SHORT_ENTRY_WORDS:
        return signals
    t = text.lower()
    signals["stressors"] = [s for s in signals["stressors"] if re.search(STRESSOR_HINTS.get(s, "^$"), t)]
    signals["positives"] = []
    signals["reflection"] = "Thanks for checking in. Short days count too."
    return signals


# Title and scene leave the private boundary (title -> Tinybird, scene -> FLUX), so they must not echo
# the entry. The prompt asks for this; a small model doesn't always comply, so enforce it here.
INDOOR = {"room", "desk", "office", "house", "home", "kitchen", "bed", "bedroom", "window", "wall", "door",
          "table", "chair", "car", "street", "city", "screen", "laptop", "phone", "meeting", "building"}
STOPWORDS = {"about", "after", "again", "being", "could", "every", "from", "have", "just", "like", "more",
             "much", "only", "some", "that", "then", "there", "this", "today", "very", "were", "what",
             "when", "with", "would", "your", "feel", "felt", "light", "still", "quiet", "little", "long"}
FALLBACK_ART = {
    "low": ("Fog over still water", "Low fog drifting over a dark, still lake with one faint light far away."),
    "mid": ("Drift and settle", "Soft layered hills under a pale, even sky at dusk."),
    "high": ("Morning opens wide", "Golden light pouring across an open meadow under a wide sky."),
}


def _words(t: str) -> set:
    return {w for w in re.findall(r"[a-z]+", t.lower()) if len(w) > 3 and w not in STOPWORDS}


def scrub_art_fields(text: str, signals: dict) -> dict:
    entry = {w.rstrip("s") for w in _words(text)}
    art = {w.rstrip("s") for w in _words(signals["title"] + " " + signals["scene"])} | \
          set(re.findall(r"[a-z]+", (signals["title"] + " " + signals["scene"]).lower()))
    if (art & entry) or (art & INDOOR):
        m = signals["mood"]
        signals["title"], signals["scene"] = FALLBACK_ART["low" if m <= 3 else "mid" if m <= 6 else "high"]
    return signals


async def _liquid_chat(payload: dict) -> str:
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(f"{LIQUID_BASE_URL}/chat/completions",
                         headers={"Authorization": f"Bearer {LIQUID_API_KEY}"}, json=payload)
        if r.status_code in (400, 422) and "response_format" in payload:
            # endpoint doesn't support schema-constrained output; fall back to prompt-only JSON
            payload.pop("response_format")
            r = await c.post(f"{LIQUID_BASE_URL}/chat/completions",
                             headers={"Authorization": f"Bearer {LIQUID_API_KEY}"}, json=payload)
        r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


async def extract_signals(text: str) -> dict:
    payload = {
        "model": LIQUID_MODEL,
        "temperature": 0.2,
        "max_tokens": 600,
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "mood_signals", "schema": EXTRACT_SCHEMA}},
        "messages": [
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": f"Journal entry:\n{text}"},
        ],
    }
    content = await _liquid_chat(payload)
    try:
        signals = normalize(_parse_json(content))
    except (ValueError, json.JSONDecodeError):
        # Small models occasionally slip; one retry with a nudge
        payload["messages"].append({"role": "assistant", "content": content})
        payload["messages"].append({"role": "user", "content": "Return only the JSON object."})
        signals = normalize(_parse_json(await _liquid_chat(payload)))
    return scrub_art_fields(text, ground_short_entry(text, signals))


# ---------- Liquid audio: private voice check-ins ----------
async def audio_available() -> bool:
    """The audio server has no /health route; any HTTP answer (even 404) means it is up."""
    try:
        async with httpx.AsyncClient(timeout=2) as c:
            await c.get(LIQUID_AUDIO_URL)
        return True
    except Exception:  # noqa: BLE001
        return False


async def transcribe(wav: bytes) -> str:
    payload = {"model": "LFM2.5-Audio-1.5B", "max_tokens": 512, "stream": True, "messages": [
        {"role": "system", "content": "Perform ASR."},
        {"role": "user", "content": [{"type": "input_audio", "input_audio": {
            "data": base64.b64encode(wav).decode(), "format": "wav"}}]}]}
    parts = []
    async with httpx.AsyncClient(timeout=120) as c:
        async with c.stream("POST", f"{LIQUID_AUDIO_URL}/chat/completions", json=payload) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data:") or line.strip() == "data: [DONE]":
                    continue
                for ch in json.loads(line[5:]).get("choices", []):
                    parts.append((ch.get("delta") or {}).get("content") or "")
    return "".join(parts).strip()


# ---------- Black Forest Labs: mood painting ----------
async def paint(signals: dict, entry_id: str) -> str:
    """Returns a local /static path, or '' if painting is unavailable."""
    body = {"prompt": art_prompt(signals), "width": 1024, "height": 1024}
    return await _flux(body, PAINT_DIR, entry_id)


async def compose_exhibition(image_paths: list, prompt: str, name: str) -> str:
    """FLUX.2 multi-reference edit: up to 8 of the week's paintings become one panorama.
    Only finished paintings and a mood-arc prompt are sent, never entry text."""
    body = {"prompt": prompt, "width": 1536, "height": 768}
    for i, p in enumerate(image_paths[:8]):
        body["input_image" if i == 0 else f"input_image_{i + 1}"] = base64.b64encode(Path(p).read_bytes()).decode()
    return await _flux(body, EXHIBIT_DIR, name)


async def _flux(body: dict, out_dir: Path, name: str) -> str:
    """Runs one FLUX job with one retry on transient errors. Returns a /static path or ''."""
    if not BFL_API_KEY:
        return ""
    for attempt in range(2):
        try:
            return await _flux_once(body, out_dir, name)
        except _Moderated as e:
            print(f"[flux] stopped with status {e}")
            return ""
        except Exception as e:  # noqa: BLE001
            print(f"[flux] attempt {attempt + 1}: {e!r}")
    return ""


class _Moderated(Exception):
    pass


async def _flux_once(body: dict, out_dir: Path, name: str) -> str:
    headers = {"x-key": BFL_API_KEY, "accept": "application/json"}
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(f"{BFL_BASE}/{BFL_MODEL}", headers=headers, json=body)
        r.raise_for_status()
        polling_url = r.json()["polling_url"]
        for _ in range(120):
            await asyncio.sleep(1)
            p = (await c.get(polling_url, headers=headers)).json()
            status = p.get("status")
            if status == "Ready":
                # BFL result URLs expire quickly, so download immediately
                img = await c.get(p["result"]["sample"])
                img.raise_for_status()
                ext = "png" if "png" in img.headers.get("content-type", "") else "jpg"
                path = out_dir / f"{name}.{ext}"
                path.write_bytes(img.content)
                return f"/static/{out_dir.name}/{path.name}"
            if status in ("Content Moderated", "Request Moderated"):
                raise _Moderated(status)
            if status in ("Error", "Failed"):
                raise RuntimeError(f"status {status}")
    raise TimeoutError("FLUX did not finish in 120s")


# ---------- Nimble: vetted, current resources ----------
TRUSTED_DOMAINS = ["nimh.nih.gov", "nhs.uk", "mayoclinic.org", "apa.org", "cdc.gov",
                   "helpguide.org", "mind.org.uk", "samhsa.gov", "sleepfoundation.org"]
TOPIC_QUERIES = {
    "work": "evidence-based ways to manage work stress and burnout",
    "school": "coping with academic stress students tips",
    "relationships": "coping with relationship stress mental health",
    "family": "managing family stress and conflict wellbeing",
    "health": "coping with stress about physical health",
    "money": "coping with financial stress and anxiety",
    "sleep": "how to improve sleep when stressed evidence-based",
    "exercise": "benefits of a daily walk for mood evidence-based",
    "loneliness": "how to cope with loneliness evidence-based tips",
    "news": "coping with anxiety from the news",
    "general": "simple evidence-based self-care for stress and low mood",
}
_resource_cache: dict[str, list] = {}


async def find_resources(topic: str) -> list[dict]:
    topic = topic if topic in TOPIC_QUERIES else "general"
    if topic in _resource_cache:
        return _resource_cache[topic]
    if not NIMBLE_API_KEY:
        return []
    try:
        async with httpx.AsyncClient(timeout=25) as c:
            r = await c.post(f"{NIMBLE_BASE}/v2/search",
                             headers={"Authorization": f"Bearer {NIMBLE_API_KEY}"},
                             json={"query": TOPIC_QUERIES[topic], "include_domains": TRUSTED_DOMAINS,
                                   "search_depth": "lite", "max_results": 4,
                                   "country": "US", "locale": "en"})
            r.raise_for_status()
        results = [{"title": x.get("title", ""), "url": x.get("url", ""),
                    "description": (x.get("description") or "")[:220]}
                   for x in r.json().get("results", []) if x.get("url")]
        _resource_cache[topic] = results
        return results
    except Exception as e:  # noqa: BLE001
        print(f"[nimble] {e}")
        return []


# ---------- Tinybird: signals store + analytics endpoints ----------
async def tb_ingest(rows: list[dict], name: str = "mood_entries") -> None:
    if name == "mood_entries":
        with MIRROR.open("a") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    body = "\n".join(json.dumps(r) for r in rows)
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.post(f"{TB_HOST}/v0/events", params={"name": name},
                         headers={"Authorization": f"Bearer {TB_TOKEN}"}, content=body)
        r.raise_for_status()


async def tb_pipe(name: str, **params) -> list[dict]:
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.get(f"{TB_HOST}/v0/pipes/{name}.json", params=params,
                        headers={"Authorization": f"Bearer {TB_TOKEN}"})
        r.raise_for_status()
    return r.json().get("data", [])


async def tb_truncate(name: str) -> None:
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.post(f"{TB_HOST}/v0/datasources/{name}/truncate",
                         headers={"Authorization": f"Bearer {TB_TOKEN}"})
        r.raise_for_status()


def mirror_rows() -> list[dict]:
    if not MIRROR.exists():
        return []
    return [json.loads(line) for line in MIRROR.read_text().splitlines() if line.strip()]


# ---------- Agent planner LLM (aggregates only) ----------
async def planner_json(system: str, user: str) -> dict:
    payload = {"model": PLANNER_MODEL, "temperature": 0.2, "max_tokens": 400,
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(f"{PLANNER_BASE_URL}/chat/completions",
                         headers={"Authorization": f"Bearer {PLANNER_API_KEY}"}, json=payload)
        r.raise_for_status()
    return _parse_json(r.json()["choices"][0]["message"]["content"])
