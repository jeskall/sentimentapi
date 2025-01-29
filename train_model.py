from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer
from datasets import load_from_disk

def train_model():
    # 1. Ladda det tokeniserade datasetet
    tokenized_datasets = load_from_disk("./tokenized_datasets")
    print("Tokenized dataset loaded!")

    # **Minska datasetets storlek för snabbare träning**
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))  # 1000 exempel
    small_test_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(500))    # 500 exempel

    # 2. Ladda en förtränad modell och tokenizer
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # 3. Definiera träningsparametrar
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",  # Byt till eval_strategy om du vill undvika varningar
        learning_rate=2e-5,
        per_device_train_batch_size=32,  # Öka till 32 om datorn klarar det
        num_train_epochs=3,
        weight_decay=0.01,
    )

    # 4. Skapa en Trainer-instans
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,  # Använd mindre dataset
        eval_dataset=small_test_dataset,    # Använd mindre dataset
    )

    # 5. Starta träningen
    print("Starting training...")
    trainer.train()
    print("Training complete!")

    # 6. Spara den tränade modellen och tokenizern
    model.save_pretrained("./sentiment_model")
    tokenizer.save_pretrained("./sentiment_model")
    print("Model and tokenizer saved in './sentiment_model'!")

if __name__ == "__main__":
    train_model()
