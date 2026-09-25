"""Prompts and fixed vocabularies. Keeping the vocab closed makes Tinybird analytics clean."""

STRESSORS = ["work", "school", "relationships", "family", "health", "money",
             "sleep", "loneliness", "news", "none"]
POSITIVES = ["exercise", "nature", "friends", "family", "rest", "accomplishment",
             "creativity", "music", "food", "none"]

EXTRACT_SYSTEM = f"""You read a short private journal entry and return ONLY a JSON object.
No prose, no markdown, no code fences.

Schema:
{{
  "mood": integer 1-10 (1 = very low, 10 = great),
  "energy": integer 1-10,
  "anxiety": integer 1-10 (1 = calm, 10 = very anxious),
  "sleep_hours": number or null (only if the entry mentions sleep),
  "emotions": up to 3 single lowercase words, e.g. ["tired", "hopeful"],
  "stressors": up to 3 from {STRESSORS},
  "positives": up to 3 from {POSITIVES},
  "risk_flag": true only if the entry mentions suicide, self-harm, wanting to die, or being unsafe,
  "title": a 2-5 word poetic title for today's painting, no names,
  "palette": 3 hex colors that express the feeling,
  "scene": one sentence describing an ABSTRACT landscape or texture that mirrors the feeling. No people, no faces, no text, no details from the entry,
  "reflection": one warm, non-clinical sentence reflecting the entry back. Never diagnose. Never give medical advice.
}}

Rules:
- Only tag stressors and positives the entry actually mentions. Map activities to the closest tag:
  any sport, walk, gym or climbing = "exercise"; making art, music, writing or crafts = "creativity";
  talking to or seeing a parent, sibling, partner or child = "family"; friends or colleagues socially = "friends".
- If the entry is very short or vague (e.g. "meh", "ok"), use mood 5, energy 5, anxiety 5, empty stressors
  and positives, and a reflection that only acknowledges the check-in. Never invent details.
- The reflection may only mention things the entry says.
- "title" and "scene" are shown to an image model and must reveal NOTHING about the entry: no places,
  rooms, objects, people, activities or events from it (no "meeting room", "office", "park", "desk").
  Use only landscape, sky, weather, water, light and texture as metaphors for the feeling.

Example output:
{{"mood": 4, "energy": 3, "anxiety": 7, "sleep_hours": 5, "emotions": ["drained", "restless"],
"stressors": ["work", "sleep"], "positives": ["nature"], "risk_flag": false,
"title": "Fog over still water", "palette": ["#5B6C8F", "#A7B3C9", "#E8C39E"],
"scene": "Low fog drifting over a grey lake with one warm light on the far shore.",
"reflection": "It sounds like a heavy, tiring day, and you still found a moment outside for yourself."}}"""


# Constrained decoding: llama-server (and most OpenAI-compatible servers) turn this into a grammar,
# so a small model cannot emit broken JSON or tags outside the closed vocabularies.
EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "mood": {"type": "integer", "minimum": 1, "maximum": 10},
        "energy": {"type": "integer", "minimum": 1, "maximum": 10},
        "anxiety": {"type": "integer", "minimum": 1, "maximum": 10},
        "sleep_hours": {"type": ["number", "null"]},
        "emotions": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        "stressors": {"type": "array", "items": {"enum": STRESSORS}, "maxItems": 3},
        "positives": {"type": "array", "items": {"enum": POSITIVES}, "maxItems": 3},
        "risk_flag": {"type": "boolean"},
        "title": {"type": "string"},
        "palette": {"type": "array", "items": {"type": "string", "pattern": "^#[0-9A-Fa-f]{6}$"},
                    "minItems": 3, "maxItems": 3},
        "scene": {"type": "string"},
        "reflection": {"type": "string"},
    },
    "required": ["mood", "energy", "anxiety", "sleep_hours", "emotions", "stressors", "positives",
                 "risk_flag", "title", "palette", "scene", "reflection"],
}


def art_prompt(s: dict) -> str:
    """Turn mood signals (never the raw text) into a FLUX prompt."""
    mood, energy = s["mood"], s["energy"]
    if mood <= 3:
        style = "heavy muted oil impasto, soft shadows, quiet and still"
    elif mood <= 6:
        style = "layered gouache, gentle diffused light, calm balance"
    else:
        style = "luminous watercolor washes, glowing light, open airy space"
    motion = ("sweeping energetic brushstrokes" if energy >= 7
              else "slow, soft blended strokes" if energy <= 3
              else "measured, steady brushwork")
    palette = ", ".join(s["palette"])
    return (f"Abstract expressive painting. {s['scene']} Style: {style}, {motion}. "
            f"Color palette: {palette}. Fine art canvas texture, square composition. "
            f"No people, no faces, no text, no letters, no signatures.")


def _mood_word(m: float) -> str:
    return "heavy and muted" if m < 4 else "unsettled" if m < 5.5 else "steady" if m < 7 else "bright and open"


def exhibition_prompt(moods: list) -> str:
    """A week's mood arc (numbers only, in day order) -> FLUX.2 multi-reference prompt."""
    half = max(1, len(moods) // 2)
    first, last = sum(moods[:half]) / half, sum(moods[half:] or moods) / len(moods[half:] or moods)
    arc = ("the light gradually opens up" if last - first >= 0.8 else
           "clouds slowly gather" if first - last >= 0.8 else "the mood holds steady")
    return (f"Combine these paintings into one gallery-quality abstract panorama, read left to right as one week. "
            f"It begins {_mood_word(first)} and ends {_mood_word(last)}; across it, {arc}. "
            f"Keep their palettes and brushwork, blended into a single continuous landscape. "
            f"No frames, no borders, no text, no people.")
