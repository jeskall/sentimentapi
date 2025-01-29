from transformers import pipeline

def test_model():
    # Ladda den tränade modellen
    sentiment = pipeline("sentiment-analysis", model="./sentiment_model")

    # Exempeltexter att testa
    examples = [
        "I absolutely love this product! It's amazing!",  # Positiv
        "This is the worst experience I've ever had.",  # Negativ
        "The movie was okay, not great but not terrible either.",  # Neutral/oklart
        "What a fantastic and wonderful day!",  # Positiv
        "I hated the service at this restaurant.",  # Negativ
        "The performance was mediocre at best.",  # Neutral/oklart
        "The package arrived damaged and late.",  # Negativ
        "I'm so happy with the results of this project!",  # Positiv
        "Not bad, but could be better.",  # Neutral/oklart
        "Terrible experience, would not recommend to anyone.",  # Negativ
    ]

    # Konvertera LABEL_0 och LABEL_1 till POSITIVE/NEGATIVE
    label_map = {0: "NEGATIVE", 1: "POSITIVE"}

    # Testa varje text
    for text in examples:
        result = sentiment(text)
        sentiment_label = label_map[int(result[0]["label"][-1])]  # Hämta POSITIVE/NEGATIVE
        confidence = result[0]["score"]
        print(f"Text: {text}")
        print(f"Sentiment: {sentiment_label}, Confidence: {confidence:.2f}\n")

if __name__ == "__main__":
    test_model()
