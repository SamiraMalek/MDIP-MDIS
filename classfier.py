import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate
import torch

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load your dataset
train_path = '/path/to/train_data.csv'
train_df = pd.read_csv(train_path)  # Replace with your dataset file
val_path = '/path/to/val_data.csv'
val_df = pd.read_csv(val_path)  # Replace with your validation dataset file

label_map = {'real': 1, 'fake': 0}
train_df['label'] = train_df['label'].map(label_map)
val_df['label'] = val_df['label'].map(label_map)

# Convert to Hugging Face dataset format
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Create a dataset dictionary
dataset_dict = DatasetDict({
    'train': train_dataset,
    'test': val_dataset
})

# Prepare and tokenize dataset
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42)
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42)

# Setup evaluation 
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Load pretrained model and evaluate model after each epoch
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
model.to(device)  # Move model to GPU if available

training_args = TrainingArguments(
    output_dir='/path/to/results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# Save the model and the tokenizer
model_save_path = "/path/to/model"
tokenizer_save_path = "/path/to/tokenizer"

model.save_pretrained(model_save_path)
tokenizer.save_pretrained(tokenizer_save_path)
