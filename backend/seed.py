"""Seed a 2-week baseline and reset the agent so the demo can fast-forward through experiments.

  python seed.py              # 14 baseline days, 6 weeks back; signals only
  python seed.py --paint 3    # also paint the last 3 baseline days with FLUX

The same synth_day() simulates experiment weeks when the agent's "Run next week" button is used.
Planted ground truth: short sleep drives low mood. Work stress co-occurs with short sleep
(workday mornings), so "work" looks like the culprit at first. Sleep is only mentioned in ~40%
of check-ins until the agent starts asking about it.
"""
import argparse
import asyncio
import datetime as dt
import os
import random
import subprocess
import uuid
from pathlib import Path

import services as S

TITLES_LOW = ["Fog over still water", "Grey hour", "Weight of the tide", "Rain on slate"]
TITLES_MID = ["Quiet middle ground", "Half-lit room", "Drift and settle", "Evening in between"]
TITLES_HIGH = ["Morning opens wide", "Warm field light", "Lifted", "Sun through linen"]
SCENES_LOW = ["Low fog over a dark lake with one faint light far away.",
              "Heavy clouds pressing on a flat grey horizon."]
SCENES_MID = ["Soft hills under a pale, even sky.", "Layered bands of sand and water at dusk."]
SCENES_HIGH = ["Golden light pouring across an open meadow.", "Bright sky over rolling green waves."]

BASELINE_DAYS = 14
SIM_WEEKS = 4


def sim_start() -> dt.date:
    return dt.date.today() - dt.timedelta(days=BASELINE_DAYS + 7 * SIM_WEEKS)


def synth_day(day: dt.date, rng: random.Random, active: str = "", habits=(), adherence: float = 0.8) -> dict:
    """One simulated check-in. `active` is the driver under test, `habits` are confirmed ones kept up."""
    dow = day.isoweekday()
    workday = dow <= 5
    done = bool(active) and rng.random() < adherence
    on = {h for h in habits if rng.random() < adherence} | ({active} if done else set())

    sleep = rng.uniform(4.6, 7.0) if workday else rng.uniform(6.5, 8.5)
    if "sleep" in on:
        sleep = rng.uniform(7.0, 8.3)
    sleep = round(sleep, 1)
    work = workday and rng.random() < (0.25 if "work" in on else 0.8)
    exercised = "exercise" in on or rng.random() < 0.35
    friends = "social" in on or rng.random() < 0.3

    mood = 5 + (sleep - 6) * 1.2 + (0.4 if exercised else 0) + (0.2 if friends else 0) \
        - (0.2 if work else 0) + rng.uniform(-0.8, 0.8)
    mood = max(1, min(10, round(mood)))
    anxiety = max(1, min(10, 4 + (2 if work else 0) + (1 if sleep < 6 else 0) + rng.randint(-1, 1)))
    energy = max(1, min(10, round(mood + rng.uniform(-2, 1.5))))
    # People mention sleep only sometimes, unless the agent's check-in prompt asks for it
    asks_sleep = "sleep" in habits or active == "sleep"
    reported = sleep if rng.random() < (0.95 if asks_sleep else 0.4) else None

    stressors = (["work"] if work else []) + (["sleep"] if reported is not None and sleep < 6 else [])
    if rng.random() < 0.15:
        stressors.append(rng.choice(["money", "relationships", "family"]))
    bucket = "low" if mood <= 3 else "mid" if mood <= 6 else "high"
    return {
        "mood": mood, "energy": energy, "anxiety": anxiety, "sleep_hours": reported,
        "emotions": rng.sample({"low": ["drained", "heavy", "restless"], "mid": ["okay", "tired", "steady"],
                                "high": ["hopeful", "light", "grateful"]}[bucket], 2),
        "stressors": stressors,
        "positives": (["exercise"] if exercised else []) + (["friends"] if friends else []),
        "palette": S._default_palette(mood),
        "title": rng.choice({"low": TITLES_LOW, "mid": TITLES_MID, "high": TITLES_HIGH}[bucket]),
        "scene": rng.choice({"low": SCENES_LOW, "mid": SCENES_MID, "high": SCENES_HIGH}[bucket]),
        "exp_id": active, "exp_done": int(done),
    }


async def simulate_days(start: dt.date, n: int, active: str = "", habits=(), paint_last: int = 0,
                        spread: bool = False) -> list[dict]:
    """Generate, optionally paint, and ingest n simulated days. Returns the rows."""
    days = []
    for i in range(n):
        day = start + dt.timedelta(days=i)
        rng = random.Random(f"{day.isoformat()}|{active}|{','.join(sorted(habits))}")
        days.append((day, rng, synth_day(day, rng, active, habits), uuid.uuid4().hex[:12]))
    # paint the last few days in parallel so a simulated week doesn't take minutes
    k = min(paint_last, n)
    # spread=True paints days evenly across the range (a week reads left to right in its exhibition)
    idx = {int((j + 0.5) * n / k) for j in range(k)} if spread and k else set(range(n - k, n))
    todo = [(d, s, eid) for i, (d, _, s, eid) in enumerate(days) if i in idx]
    images = dict(zip([eid for *_, eid in todo], await asyncio.gather(*(S.paint(s, eid) for _, s, eid in todo))))
    rows = []
    for day, rng, s, entry_id in days:
        ts = dt.datetime.combine(day, dt.time(21, rng.randint(0, 59)))
        rows.append({
            "entry_id": entry_id, "user_id": S.USER_ID, "ts": ts.strftime("%Y-%m-%d %H:%M:%S"),
            **{k: s[k] for k in ("mood", "energy", "anxiety", "sleep_hours", "emotions", "stressors",
                                 "positives", "palette", "title", "exp_id", "exp_done")},
            "risk_flag": 0, "image_path": images.get(entry_id, ""),
        })
    try:
        await S.tb_ingest(rows)
    except Exception as e:  # noqa: BLE001
        print(f"[tinybird] ingest failed, local mirror only: {e}")
    return rows


async def truncate(name: str) -> None:
    """The app token is append-only, so fall back to the logged-in tb CLI. Never seed on top of old data."""
    try:
        await S.tb_truncate(name)
        return
    except Exception:  # noqa: BLE001
        pass
    # drop the app's TB_* vars so tb uses the admin login in tinybird/.tinyb, not the append-only token
    env = {k: v for k, v in os.environ.items() if not k.startswith("TB_")}
    env["PATH"] = f"{Path.home() / '.local/bin'}:{env.get('PATH', '')}"
    tb_dir = Path(__file__).parent.parent / "tinybird"
    r = subprocess.run(["tb", "--cloud", "datasource", "truncate", name, "--yes"], cwd=tb_dir, env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"Could not truncate {name}. Run `tb login` in tinybird/ first.\n{r.stdout}{r.stderr}")


async def main(paint_last: int):
    import agent
    for name in ("mood_entries", "agent_events"):
        await truncate(name)
    S.MIRROR.unlink(missing_ok=True)
    rows = await simulate_days(sim_start(), BASELINE_DAYS, paint_last=paint_last)
    for r in rows:
        print(f"{r['ts'][:10]}  mood {r['mood']}  anxiety {r['anxiety']}  sleep {r['sleep_hours']}  "
              f"{'painted' if r['image_path'] else ''}")
    agent.reset()
    print(f"Seeded {len(rows)} baseline days from {sim_start()}. Agent reset; use 'Run next week' in the UI.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--paint", type=int, default=0)
    a = ap.parse_args()
    asyncio.run(main(a.paint))
