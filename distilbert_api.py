import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from fastapi.middleware.cors import CORSMiddleware

# -------- Config via environment variables --------
MODEL_ID = os.getenv("MODEL_ID", "distilbert-base-uncased")  # change to your HF repo if you pushed your fine-tuned model
HF_TOKEN = os.getenv("HF_TOKEN")  # optional, only if your HF repo is private

app = FastAPI(title="DistilBERT Priority API")

# CORS (let your Android app call it)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock down later if you want
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Load model once on startup --------
print(f"Loading model: {MODEL_ID}")
tok = DistilBertTokenizerFast.from_pretrained(MODEL_ID, use_auth_token=HF_TOKEN)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_ID, use_auth_token=HF_TOKEN)
model.eval()
id2label = model.config.id2label  # e.g. {0:'Low',1:'Medium',2:'High'}

class PredictIn(BaseModel):
    text: str

class PredictOut(BaseModel):
    priority: str
    raw: dict

@app.get("/")
def root():
    return {"ok": True, "model": MODEL_ID, "labels": id2label}

@app.post("/predict", response_model=PredictOut)
def predict(body: PredictIn):
    text = (body.text or "").strip()
    if not text:
        return {"priority": "Medium", "raw": {"reason": "empty"}}

    inputs = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
        pred_id = int(torch.argmax(logits, dim=-1).item())
    label = id2label.get(pred_id, "Medium")

    # Normalize to your app’s labels
    if label.lower().startswith("hi"):
        label = "High"
    elif label.lower().startswith("lo"):
        label = "Low"
    else:
        label = "Medium"

    return {"priority": label, "raw": {"pred_id": pred_id}}
