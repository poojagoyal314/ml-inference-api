from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import json
from datetime import datetime, timezone
from pathlib import Path

app = FastAPI()
model = joblib.load("model.joblib")

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "predictions.log"

class Features(BaseModel):
    values: list[float] = Field(..., min_length=4, max_length=4, examples=[[5.1, 3.5, 1.4, 0.2]])

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(f: Features):
    pred = int(model.predict([f.values])[0])
    entry = {
        "time": datetime.now(timezone.utc).isoformat(),
        "input": f.values,
        "prediction": pred,
    }
    with LOG_FILE.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")
    return {"prediction": pred}

