from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer

def prepare_data():
    # 1. Ladda dataset
    dataset = load_dataset("imdb")  # Byt till annat dataset om du vill
    print("Dataset loaded:", dataset)

    # 2. Ladda tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # 3. Tokenisera data
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # 4. Spara tokeniserad data
    tokenized_datasets.save_to_disk("./tokenized_datasets")
    print("Tokenized datasets saved!")
    return tokenized_datasets

if __name__ == "__main__":
    prepare_data()

tokenized_datasets = load_from_disk("./tokenized_datasets")
print("Sample from tokenized dataset:", tokenized_datasets["train"][0])
