import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from typing import List, Union
from fastapi.responses import JSONResponse

# Create FastAPI app
app = FastAPI()

# Load the trained model from Hugging Face
sentiment = pipeline("sentiment-analysis", model="jeskall/sentiment-model")

# Define input model
class TextInput(BaseModel):
    text: Union[str, None] = None  # Single text input
    texts: Union[List[str], None] = None  # Batch input

# ✅ Root Endpoint (Health Check)
@app.get("/")
def read_root():
    return {"message": "Welcome to the Sentiment Analysis API!"}

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API is running!"}

# 📌 Single Sentiment Analysis
@app.post("/analyze/single")
def analyze_single(input: TextInput):
    if not input.text:
        raise HTTPException(status_code=400, detail="Please provide a valid 'text' field.")
    
    result = sentiment(input.text)
    label_map = {"LABEL_0": "NEGATIVE", "LABEL_1": "POSITIVE"}
    sentiment_label = label_map[result[0]["label"]]

    return {
        "text": input.text,
        "sentiment": sentiment_label,
        "confidence": result[0]["score"]
    }

# 📌 Batch Sentiment Analysis (Max 10 Texts)
@app.post("/analyze/batch")
def analyze_batch(input: TextInput):
    if not input.texts or len(input.texts) == 0:
        raise HTTPException(status_code=400, detail="Please provide a valid 'texts' list.")
    
    if len(input.texts) > 10:
        raise HTTPException(status_code=400, detail="Maximum of 10 texts allowed in batch processing.")

    results = []
    for text in input.texts:
        result = sentiment(text)
        label_map = {"LABEL_0": "NEGATIVE", "LABEL_1": "POSITIVE"}
        sentiment_label = label_map[result[0]["label"]]

        results.append({
            "text": text,
            "sentiment": sentiment_label,
            "confidence": result[0]["score"]
        })

    response = JSONResponse(content={"total_texts_analyzed": len(input.texts), "results": results})
    response.headers["X-RapidAPI-Quota-Used"] = str(len(input.texts))  # Track quota in RapidAPI

    return response

# 📌 Premium Batch Sentiment Analysis (Max 100 Texts)
@app.post("/analyze/premium")
def analyze_premium(input: TextInput):
    if not input.texts or len(input.texts) == 0:
        raise HTTPException(status_code=400, detail="Please provide a valid 'texts' list.")
    
    if len(input.texts) > 100:
        raise HTTPException(status_code=400, detail="Maximum of 100 texts allowed in premium batch processing.")

    results = []
    for text in input.texts:
        result = sentiment(text)
        label_map = {"LABEL_0": "NEGATIVE", "LABEL_1": "POSITIVE"}
        sentiment_label = label_map[result[0]["label"]]

        results.append({
            "text": text,
            "sentiment": sentiment_label,
            "confidence": result[0]["score"]
        })

    response = JSONResponse(content={"total_texts_analyzed": len(input.texts), "results": results})
    response.headers["X-RapidAPI-Quota-Used"] = str(len(input.texts))  # Track premium quota usage

    return response

# 🚀 Ensure the app binds to the correct port for Render
if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))  # Default to 10000 locally
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
