from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib

app = FastAPI()
model = joblib.load("model.joblib")

class Features(BaseModel):
    values: list[float] = Field(
        ...,
        min_length=4,                                        # adding checks to ensure valid data
        max_length=4,
        examples=[[5.1, 3.5, 1.4, 0.2]]
    )


@app.get("/health")
def health():
    return {"status": "OK"}


@app.post("/predict")
def predict(f:Features):
    pred = model.predict([f.values])
    return{"prediction": int(pred[0])}

