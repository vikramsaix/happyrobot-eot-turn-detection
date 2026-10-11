"""Deterministic toy corpus generator; examples are synthetic, not call data."""
from __future__ import annotations

import csv
import random
from pathlib import Path

SCENARIOS = [
    # English: alternating turn-complete / continuation snapshots.
    ("en", "complete", "I need to reschedule my appointment."),
    ("en", "open", "I need to reschedule my appointment because"),
    ("en", "complete", "Could you send it tomorrow?"),
    ("en", "open", "Could you send it"),
    ("en", "complete", "That's all I needed, thanks."),
    ("en", "open", "That's all I needed, but"),
    ("en", "complete", "Yes, that works for me."),
    ("en", "open", "Yes, that works for me if"),
    ("en", "complete", "The address is forty-two Market Street."),
    ("en", "open", "The address is forty-two Market"),
    ("en", "complete", "Okay, goodbye."),
    ("en", "open", "Okay, and another thing"),
    ("en", "complete", "I'm calling about my order."),
    ("en", "open", "I'm calling about my order and"),
    ("en", "complete", "Please call me back tomorrow."),
    ("en", "open", "Please call me back when"),
    ("en", "complete", "I would like to cancel the subscription."),
    ("en", "open", "I would like to cancel the"),
    ("en", "complete", "No, that is not correct."),
    ("en", "open", "No, that is not"),
    ("en", "complete", "Let me know if you need anything else."),
    ("en", "open", "Let me know if you need"),
    ("en", "complete", "Thank you for your help."),
    ("en", "open", "Thank you for your help, and"),
    # Spanish bonus: narrow lexical coverage, not a claim of multilingual readiness.
    ("es", "complete", "Eso es todo, muchas gracias."),
    ("es", "open", "Eso es todo, pero"),
    ("es", "complete", "¿Podrías enviarlo mañana?"),
    ("es", "open", "¿Podrías enviarlo"),
    ("es", "complete", "Sí, eso funciona para mí."),
    ("es", "open", "Sí, eso funciona para mí si"),
    ("es", "complete", "La dirección es Calle Mayor cuarenta y dos."),
    ("es", "open", "La dirección es Calle Mayor"),
    ("es", "complete", "De acuerdo, hasta luego."),
    ("es", "open", "De acuerdo, y otra cosa"),
    ("es", "complete", "Quiero cancelar la suscripción."),
    ("es", "open", "Quiero cancelar la"),
    ("es", "complete", "Gracias por tu ayuda."),
    ("es", "open", "Gracias por tu ayuda, porque"),
]


def build_rows(seed: int = 731, per_scenario: int = 80) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    for sid, (lang, kind, text) in enumerate(SCENARIOS):
        for sample_id in range(per_scenario):
            # Stratified signal ranges ensure the baseline sees ambiguous pauses too.
            if sample_id % 4 == 0:
                pause = rng.randint(40, 220)
            elif sample_id % 4 == 1:
                pause = rng.randint(220, 520)
            elif sample_id % 4 == 2:
                pause = rng.randint(520, 1050)
            else:
                pause = rng.randint(1050, 1850)
            stability = rng.randint(30, 850)
            # Completion intent is a synthetic label prior; longer silences help, but
            # short pauses rarely mean the agent should seize the floor.
            if kind == "complete":
                p_end = 0.12 if pause < 180 else (0.74 if pause < 450 else 0.94)
            else:
                p_end = 0.015 if pause < 900 else (0.10 if pause < 1450 else 0.25)
            if stability < 100:
                p_end *= 0.2
            label = int(rng.random() < p_end)
            rows.append({
                "scenario_id": sid,
                "scenario_kind": kind,
                "language": lang,
                "text": text,
                "pause_ms": pause,
                "stability_ms": stability,
                "label_respond": label,
            })
    return rows


def write_csv(path: str | Path, rows: list[dict]) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
