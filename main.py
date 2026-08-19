from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import os
import json
import logging
from pathlib import Path
from sqlalchemy import create_engine, text

app = FastAPI()
model = joblib.load("model.joblib")

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "predictions.log"

DATABASE_URL = os.environ["DATABASE_URL"]
engine = create_engine(DATABASE_URL)

@app.on_event("startup")
def setup_database():
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                created_at TIMESTAMPTZ DEFAULT now(),
                input JSONB NOT NULL,
                prediction INTEGER NOT NULL
            )
        """))

class Features(BaseModel):
    values: list[float] = Field(..., min_length=4, max_length=4, examples=[[5.1, 3.5, 1.4, 0.2]])

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(f: Features):
    pred = int(model.predict([f.values])[0])
    record = {"input": f.values, "prediction": pred}

    # Safety-net write: to the file first, and never let it fail the request.
    try:
        with LOG_FILE.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception:
        logging.exception("Failed to write prediction to log file")

    # Durable write: to Postgres. If this fails, we've still got the file record.
    try:
        with engine.begin() as conn:
            conn.execute(
                text("INSERT INTO predictions (input, prediction) VALUES (:input, :prediction)"),
                {"input": json.dumps(f.values), "prediction": pred},
            )
    except Exception:
        logging.exception("Failed to write prediction to database")

    return {"prediction": pred}