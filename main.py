from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import torch

app = FastAPI()

MODEL_NAME = "manthansubhash01/sbert-stsb-manual"
model = SentenceTransformer(MODEL_NAME)

class BatchRequest(BaseModel):
    newIdea: str
    existingIdeas: list[str]

@app.get("/")
def root():
    return {"message": "ML Similarity API is running"}

@app.get("/favicon.ico")
def favicon():
    return {}

@app.post("/predict_batch")
def predict_batch(data: BatchRequest):
    new = data.newIdea
    existing = data.existingIdeas

    enc_new = model.encode(new, convert_to_tensor=True)
    enc_existing = model.encode(existing, convert_to_tensor=True)

    similarities = util.cos_sim(enc_new, enc_existing)[0]

    similarities = similarities.cpu().tolist()

    max_score = max(similarities)

    return {
        "scores": similarities,
        "max_score": max_score,
    }