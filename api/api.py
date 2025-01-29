import os
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from typing import List, Union

# Skapa FastAPI-app
app = FastAPI()

# Ladda din tränade modell
sentiment = pipeline("sentiment-analysis", model="./sentiment_model")

# Modell för att hantera indata
class TextInput(BaseModel):
    text: Union[str, None] = None  # För en text
    texts: Union[List[str], None] = None  # För flera texter (batch)

# Root-endpoint (hälsa världen)
@app.get("/")
def read_root():
    return {"message": "Welcome to the Sentiment Analysis API!"}

# Endpoint för sentimentanalys
@app.post("/analyze")
def analyze_text(input: TextInput):
    # Kontrollera om det är en enstaka text eller batch
    if input.text:
        # En text
        result = sentiment(input.text)
        label_map = {0: "NEGATIVE", 1: "POSITIVE"}
        sentiment_label = label_map[int(result[0]["label"][-1])]
        return {
            "text": input.text,
            "sentiment": sentiment_label,
            "confidence": result[0]["score"]
        }
    elif input.texts:
        # Flera texter (batch)
        results = []
        for text in input.texts:
            result = sentiment(text)
            label_map = {0: "NEGATIVE", 1: "POSITIVE"}
            sentiment_label = label_map[int(result[0]["label"][-1])]
            results.append({
                "text": text,
                "sentiment": sentiment_label,
                "confidence": result[0]["score"]
            })
        return results
    else:
        return {"error": "You must provide either 'text' or 'texts'."}

# 🚀 Ensure the app binds to the correct port for Render
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Default to 8000 locally
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
