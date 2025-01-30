import os
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from typing import List, Union

# Create FastAPI app
app = FastAPI()

# Load the trained model from Hugging Face
sentiment = pipeline("sentiment-analysis", model="jeskall/sentiment-model")

# Define input model
class TextInput(BaseModel):
    text: Union[str, None] = None  # Single text input
    texts: Union[List[str], None] = None  # Batch input

# 🚀 Root Endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Sentiment Analysis API!"}

# ✅ API Health Check
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API is running!"}

# 📌 Single Sentiment Analysis
@app.post("/analyze/single")
def analyze_single(input: TextInput):
    if not input.text:
        return {"error": "Please provide a valid 'text' field."}
    
    result = sentiment(input.text)
    label_map = {"LABEL_0": "NEGATIVE", "LABEL_1": "POSITIVE"}  # Ensure labels match model
    sentiment_label = label_map[result[0]["label"]]

    return {
        "text": input.text,
        "sentiment": sentiment_label,
        "confidence": result[0]["score"]
    }

# 📌 Optimized Batch Sentiment Analysis
@app.post("/analyze/batch")
def analyze_batch(input: TextInput):
    if not input.texts or len(input.texts) == 0:
        return {"error": "Please provide a valid 'texts' list."}
    
    # Process all texts at once for speed improvement
    batch_results = sentiment(input.texts)
    label_map = {"LABEL_0": "NEGATIVE", "LABEL_1": "POSITIVE"}

    return [
        {
            "text": text,
            "sentiment": label_map[result["label"]],
            "confidence": result["score"]
        }
        for text, result in zip(input.texts, batch_results)
    ]

# 🌍 Unified Sentiment Analysis (Handles both single & batch)
@app.post("/analyze")
def analyze_text(input: TextInput):
    if input.text:
        return analyze_single(input)
    elif input.texts:
        return analyze_batch(input)
    else:
        return {"error": "You must provide either 'text' or 'texts'."}

# 🚀 Ensure the app binds to the correct port for Render
if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))  # Default to 10000 locally
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
