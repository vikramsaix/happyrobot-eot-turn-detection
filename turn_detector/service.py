from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .model import TurnDetector

MODEL_PATH = Path(os.environ.get("TURN_MODEL_PATH", Path(__file__).resolve().parents[1] / "artifacts" / "model.json"))
detector = TurnDetector(MODEL_PATH)
app = FastAPI(title="Turn Detection Baseline", version="1.0.0", description="Text-and-timing endpointing prototype; not an audio VAD.")


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    pause_ms: float = Field(ge=0, le=10000, description="Current no-speech duration from upstream VAD, in ms")
    stability_ms: float = Field(ge=0, le=10000, description="Time the current ASR hypothesis has remained stable, in ms")
    language: Literal["en", "en-US", "en-GB", "es", "es-ES", "es-419"] = "en"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_version": detector.model_version}


@app.post("/predict")
def predict(request: PredictRequest) -> dict:
    try:
        return detector.predict(request.text, request.pause_ms, request.stability_ms, request.language)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
