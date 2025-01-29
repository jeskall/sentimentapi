from transformers import AutoTokenizer

def save_tokenizer():
    # Ladda tokenizern från samma modell som användes för träning
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Spara tokenizern i samma mapp som modellen
    tokenizer.save_pretrained("./sentiment_model")
    print("Tokenizer saved in './sentiment_model'!")

if __name__ == "__main__":
    save_tokenizer()