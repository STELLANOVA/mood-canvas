"""The long-horizon agent: one weekly cycle of observe -> correct -> plan -> act.

The agent never re-reads journal text or its own full log. Its whole working context is the
memory card (a few hundred tokens) plus fresh Tinybird aggregates. Verdicts are rule-based so
they are reproducible; the planner LLM only picks from a closed candidate list and writes prose,
with deterministic fallbacks so a flaky model never breaks the loop.
"""
import datetime as dt
import json
import math

import services as S
import seed

STATE = S.DATA_DIR / "agent_state.json"   # the memory card: all the agent carries between cycles
LOG = S.DATA_DIR / "agent_log.jsonl"      # the lab notebook: for humans and the UI, never fed back to the agent

GOAL = "Find out what drives low-mood days and build one small habit that measurably helps."
MIN_ENTRIES = 4          # fewer check-ins than this in a window = inconclusive
SUPPORT_DELTA = 0.8      # mood points over baseline needed to call a habit helpful
LEVER_DELTA = 0.2        # the habit must actually move its lever by this much
MAX_FINDINGS = 6         # older findings are folded into notes, so the card stays small

DRIVERS = {
    "sleep": {"hypothesis": "More sleep on work nights will lift mood.", "label": "short sleep", "topic": "sleep",
              "habit": "Lights out by 23:00, phone charging outside the bedroom.",
              "smaller": "Lights out by 23:30 on work nights only.",
              "ask": "How many hours did you sleep?"},
    "work": {"hypothesis": "Closing the workday deliberately will lift mood.", "label": "work stress", "topic": "work",
             "habit": "10-minute shutdown ritual: write tomorrow's top 3, then close the laptop.",
             "smaller": "Write tomorrow's top 3 before logging off. That's it.",
             "ask": "Did you do your shutdown ritual?"},
    "exercise": {"hypothesis": "A daily walk will lift mood.", "label": "no movement", "topic": "exercise",
                 "habit": "A 20-minute walk outside before 6pm.",
                 "smaller": "A 10-minute walk, any time.",
                 "ask": "Did you get outside for a walk?"},
    "social": {"hypothesis": "A daily conversation with someone will lift mood.", "label": "no social contact", "topic": "loneliness",
               "habit": "One real conversation a day: a call, a coffee, or a voice note to a friend.",
               "smaller": "One message to a friend each day.",
               "ask": "Did you connect with someone today?"},
}
LEVER_TEXT = {"sleep": "short-sleep nights", "work": "days mentioning work stress",
              "exercise": "days without exercise", "social": "days without friends"}


# ---------- memory card ----------
def reset() -> dict:
    start = seed.sim_start()
    card = {
        "goal": GOAL,
        "cycle": 0,
        "clock": (start + dt.timedelta(days=seed.BASELINE_DAYS)).isoformat(),
        "baseline": {"start": start.isoformat(),
                     "end": (start + dt.timedelta(days=seed.BASELINE_DAYS)).isoformat()},
        "habits": [],
        "active": None,
        "findings": [],
        "notes": "",
        "paused": False,
        "new_baseline": None,
    }
    save(card)
    LOG.unlink(missing_ok=True)
    return card


def load() -> dict:
    if not STATE.exists():
        return reset()
    return json.loads(STATE.read_text())


def save(card: dict) -> None:
    STATE.write_text(json.dumps(card, indent=1))


def card_tokens(card: dict) -> int:
    return len(json.dumps(card)) // 4  # rough, good enough to show it stays flat


def read_log() -> list[dict]:
    if not LOG.exists():
        return []
    return [json.loads(line) for line in LOG.read_text().splitlines() if line.strip()]


async def log(card: dict, phase: str, title: str, detail: str = "", evidence=None) -> dict:
    ev = {"ts": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), "user_id": S.USER_ID,
          "cycle": card["cycle"], "phase": phase, "title": title, "detail": detail}
    with LOG.open("a") as f:
        f.write(json.dumps({**ev, "sim_week_of": card["clock"], "evidence": evidence}) + "\n")
    try:
        await S.tb_ingest([ev], name="agent_events")
    except Exception as e:  # noqa: BLE001
        print(f"[tinybird] agent_events: {e}")
    return ev


# ---------- observation: Tinybird first, local mirror as fallback ----------
def _ts(d: str) -> str:
    return f"{d} 00:00:00"


def _rows(start: str, end: str) -> list[dict]:
    return [r for r in S.mirror_rows()
            if r.get("user_id") == S.USER_ID and _ts(start) <= r["ts"] < _ts(end)]


def _avg(xs) -> float:
    xs = list(xs)
    return round(sum(xs) / len(xs), 2) if xs else float("nan")


def local_window(start: str, end: str) -> dict:
    rows = _rows(start, end)
    n = len(rows)
    rep = [r for r in rows if r.get("sleep_hours") is not None]
    frac = (lambda k: round(sum(1 for r in rows if k(r)) / n, 2)) if n else (lambda k: 0.0)
    return {
        "n": n, "mood": _avg(r["mood"] for r in rows), "anxiety": _avg(r["anxiety"] for r in rows),
        "energy": _avg(r["energy"] for r in rows), "sleep_reported": len(rep),
        "sleep_rate": round(sum(1 for r in rep if r["sleep_hours"] < 6) / len(rep), 2) if rep else 0.0,
        "work_rate": frac(lambda r: "work" in r["stressors"]),
        "exercise_rate": frac(lambda r: "exercise" not in r["positives"]),
        "social_rate": frac(lambda r: "friends" not in r["positives"]),
        "adherence": _avg(r.get("exp_done", 0) for r in rows) if n else 0.0,
        "risk": sum(r.get("risk_flag", 0) for r in rows),
    }


def local_scan(start: str, end: str) -> list[dict]:
    rows = _rows(start, end)
    tests = {
        "sleep": (lambda r: r.get("sleep_hours") is not None and r["sleep_hours"] < 6,
                  lambda r: r.get("sleep_hours") is not None and r["sleep_hours"] >= 6),
        "work": (lambda r: "work" in r["stressors"], lambda r: "work" not in r["stressors"]),
        "exercise": (lambda r: "exercise" not in r["positives"], lambda r: "exercise" in r["positives"]),
        "social": (lambda r: "friends" not in r["positives"], lambda r: "friends" in r["positives"]),
    }
    out = []
    for d, (yes, no) in tests.items():
        w = [r["mood"] for r in rows if yes(r)]
        out.append({"driver": d, "n_with": len(w), "mood_with": _avg(w),
                    "mood_without": _avg(r["mood"] for r in rows if no(r))})
    return out


async def window(start: str, end: str) -> tuple:
    local = local_window(start, end)
    try:
        data = await S.tb_pipe("window_stats", user_id=S.USER_ID, start=_ts(start), end=_ts(end))
        # Tinybird makes new events queryable a few seconds after ingest; if it has fewer rows than
        # the local mirror it is still catching up, so its aggregates would be wrong.
        if data and int(data[0].get("n") or 0) >= local["n"]:
            return data[0], "tinybird"
        print(f"[tinybird] window_stats behind local mirror ({(data or [{}])[0].get('n')} < {local['n']} rows)")
    except Exception as e:  # noqa: BLE001
        print(f"[tinybird] window_stats: {e}")
    return local, "local"


async def scan(start: str, end: str) -> tuple:
    _, src = await window(start, end)  # same freshness check; driver_scan has no row count of its own
    if src == "tinybird":
        try:
            data = await S.tb_pipe("driver_scan", user_id=S.USER_ID, start=_ts(start), end=_ts(end))
            if data:
                return data, "tinybird"
        except Exception as e:  # noqa: BLE001
            print(f"[tinybird] driver_scan: {e}")
    return local_scan(start, end), "local"


def _num(v) -> float:
    try:
        v = float(v)
        return v if math.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


# ---------- the cycle ----------
def verdict(base: dict, exp: dict, driver: str, retried: bool) -> tuple:
    """Rule-based and reproducible. Returns (verdict, mood_delta, lever_delta)."""
    mood_delta = round(_num(exp["mood"]) - _num(base["mood"]), 2)
    lever_delta = round(_num(base[f"{driver}_rate"]) - _num(exp[f"{driver}_rate"]), 2)
    if int(exp["n"]) < MIN_ENTRIES:
        return "too_few", mood_delta, lever_delta
    if _num(exp["adherence"]) < 0.5 and not retried:
        return "too_hard", mood_delta, lever_delta
    if not lever_delta >= LEVER_DELTA:
        return "lever_stuck", mood_delta, lever_delta
    if mood_delta >= SUPPORT_DELTA:
        return "supported", mood_delta, lever_delta
    return "rejected", mood_delta, lever_delta


async def observe_and_correct(card: dict, end: str) -> None:
    a = card["active"]
    d = a["driver"]
    (base, src_b), (exp, src_e) = await window(**a["baseline"]), await window(a["start"], end)
    await log(card, "observe", f"Measured week {card['cycle']}: {DRIVERS[d]['label']} experiment",
              f"{exp['n']} check-ins. Mood {base['mood']} → {exp['mood']}. "
              f"{LEVER_TEXT[d].capitalize()}: {round(_num(base[f'{d}_rate']) * 100)}% → "
              f"{round(_num(exp[f'{d}_rate']) * 100)}%. Plan followed on {round(_num(exp['adherence']) * 100)}% of days.",
              {"baseline": base, "experiment": exp, "source": src_e if src_e == src_b else "mixed"})

    if int(exp.get("risk") or 0):
        card["paused"] = True
        await log(card, "correct", "Paused: a check-in this week raised a safety flag",
                  "Experiments stop until you choose to resume. Crisis resources were shown at the time.")
        card["active"] = None
        return

    v, md, ld = verdict(base, exp, d, a.get("retried", False))
    rule = {
        "too_few": f"Only {exp['n']} check-ins, not enough to judge. Running the same experiment one more week.",
        "too_hard": "The plan was followed on fewer than half the days. Shrinking the ask and retrying once.",
        "lever_stuck": f"The habit didn't change {LEVER_TEXT[d]}, so this week says nothing about {DRIVERS[d]['label']}.",
        "supported": f"The lever moved and mood rose {md:+.1f}. Keeping this habit and testing the next idea on top of it.",
        "rejected": f"The lever moved ({LEVER_TEXT[d]} down {round(ld * 100)} pts) but mood changed {md:+.2f}, under the {SUPPORT_DELTA:+.1f} bar. "
                    f"{DRIVERS[d]['label'].capitalize()} looked guilty, but it isn't the driver.",
    }[v]

    # The rule text is the lesson: a small model paraphrasing it drifted from the numbers in testing.
    lesson = rule
    # running notes are written by code, so they can't drift from the data
    card["notes"] = (card["notes"] + f" W{card['cycle']}: {d} {v} (mood {md:+.2f}).").strip()[-350:]

    if v == "too_few":
        a["end"] = (dt.date.fromisoformat(end) + dt.timedelta(days=7)).isoformat()
        await log(card, "correct", "Inconclusive: extending the experiment", rule)
        return
    if v == "too_hard":
        a.update(retried=True, start=end, habit=DRIVERS[d]["smaller"],
                 end=(dt.date.fromisoformat(end) + dt.timedelta(days=7)).isoformat())
        await log(card, "correct", "Too hard to keep up: making the ask smaller", rule, {"new_habit": a["habit"]})
        return

    card["findings"].append({"driver": d, "verdict": v, "mood_delta": md, "lever_delta": ld, "lesson": lesson})
    if len(card["findings"]) > MAX_FINDINGS:  # compaction: fold the oldest finding into notes
        old = card["findings"].pop(0)
        card["notes"] = (f"Earlier: {old['driver']} {old['verdict']}. " + card["notes"])[:350]
    if v == "supported":
        card["habits"].append(d)
        # later experiments are compared against life *with* this habit, not the original baseline
        card["new_baseline"] = {"start": a["start"], "end": end}
    card["active"] = None
    title = {"supported": f"Supported: {DRIVERS[d]['label']} matters",
             "rejected": f"Rejected: {DRIVERS[d]['label']} isn't the driver",
             "lever_stuck": "Inconclusive: the habit didn't take"}[v]
    await log(card, "correct", title, lesson)


async def plan_and_act(card: dict, start: str, simulate: bool) -> None:
    tested = {f["driver"] for f in card["findings"]} | set(card["habits"])
    rows, src = await scan(card["baseline"]["start"], start)
    cands = []
    for r in rows:
        d = r["driver"]
        gap, n = _num(r["mood_without"]) - _num(r["mood_with"]), int(r["n_with"] or 0)
        if d in tested or not math.isfinite(gap) or gap <= 0:
            continue  # a driver can only be "dragging mood down" if mood is actually lower with it
        # evidence-weighted gap: a big gap seen on 2 days is weaker than a moderate one seen on 8
        cands.append({"driver": d, "days_seen": n, "mood_gap": round(gap, 2),
                      "score": round(gap * min(1.0, n / 8), 2)})
    cands.sort(key=lambda c: -c["score"])
    if not cands:
        await log(card, "plan", "No untested ideas left",
                  f"Keeping confirmed habits: {', '.join(card['habits']) or 'none'}. Watching for new patterns.",
                  {"candidates": rows, "source": src})
        if simulate:  # time still passes: simulate a maintenance week with the kept habits
            await seed.simulate_days(dt.date.fromisoformat(start), 7, habits=tuple(card["habits"]), paint_last=3, spread=True)
            card["clock"] = (dt.date.fromisoformat(start) + dt.timedelta(days=7)).isoformat()
        return

    # The ranking is the decision; the LLM may break near-ties and explains the choice in plain words.
    close = [c for c in cands if c["score"] >= 0.75 * cands[0]["score"]]
    facts = {c["driver"]: f"On {LEVER_TEXT[c['driver']]} mood averages {c['mood_gap']:.1f} points lower "
                          f"(seen on {c['days_seen']} days)." for c in close}
    pick, why = cands[0]["driver"], ""
    try:
        out = await S.planner_json(
            "You are the planning step of a wellbeing experiment agent. Pick ONE driver to test this week "
            "from the options and explain why in one plain sentence, using ONLY the facts given. "
            "Earlier results: " + (card["notes"] or "none yet") + " "
            "Never diagnose or give medical advice. Return ONLY JSON: {\"driver\": \"...\", \"why\": \"...\"}",
            json.dumps({"options": facts}))
        if out.get("driver") in facts:
            pick, why = out["driver"], str(out.get("why") or "")[:240]
    except Exception as e:  # noqa: BLE001
        print(f"[planner] plan fallback: {e}")
    c = next(x for x in cands if x["driver"] == pick)
    why = why or (f"Mood is {c['mood_gap']:.1f} points lower on {LEVER_TEXT[pick]}, "
                  f"seen on {c['days_seen']} days: the strongest untested signal.")

    end = (dt.date.fromisoformat(start) + dt.timedelta(days=7)).isoformat()
    card["active"] = {"driver": pick, "hypothesis": DRIVERS[pick]["hypothesis"],
                      "habit": DRIVERS[pick]["habit"], "ask": DRIVERS[pick]["ask"],
                      "start": start, "end": end,
                      "baseline": card.get("new_baseline") or card["baseline"]}
    await log(card, "plan", f"Hypothesis: {DRIVERS[pick]['label']} is dragging mood down", why,
              {"candidates": cands, "source": src})

    evidence = await S.find_resources(DRIVERS[pick]["topic"])
    await log(card, "act", f"Started a 7-day experiment: {card['active']['habit']}",
              f"Daily check-in question: “{DRIVERS[pick]['ask']}”",
              {"sources": evidence[:2]})
    if simulate:
        rows = await seed.simulate_days(dt.date.fromisoformat(start), 7, active=pick,
                                        habits=tuple(card["habits"]), paint_last=3, spread=True)
        await log(card, "act", "Simulated 7 check-ins for the demo",
                  f"Generated by seed.synth_day with the habit applied on ~80% of days. Moods: "
                  f"{', '.join(str(r['mood']) for r in rows)}.")
    card["clock"] = end


async def step(simulate: bool = True, resume: bool = False) -> dict:
    card = load()
    if card["paused"] and not resume:
        return {"card": card, "message": "Experiments are paused after a safety flag."}
    card["paused"] = False
    today = dt.date.today().isoformat()
    if simulate and card["clock"] >= today:
        return {"card": card, "message": "The simulation has caught up to today. New weeks now come from live check-ins."}
    now = card["clock"] if simulate else today
    card["cycle"] += 1
    if card["active"]:
        await observe_and_correct(card, now)
    if not card["paused"]:
        if card["active"]:  # extended or retried: keep running it
            if simulate:
                await seed.simulate_days(dt.date.fromisoformat(now), 7, active=card["active"]["driver"],
                                         habits=tuple(card["habits"]), paint_last=3, spread=True)
                card["clock"] = (dt.date.fromisoformat(now) + dt.timedelta(days=7)).isoformat()
        else:
            await plan_and_act(card, now, simulate)
    save(card)
    return {"card": card, "message": ""}


def pause_for_safety() -> None:
    card = load()
    card["paused"] = True
    save(card)
