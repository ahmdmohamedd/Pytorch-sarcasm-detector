import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample sarcasm dataset (replace this with a real dataset)
data = {
    "text": [
        "Oh great! Another rainy day.",
        "I just love standing in line for hours.",
        "This is the best day of my life!",
        "I'm so glad I forgot my umbrella.",
        "Really? I had no idea.",
        "Can't wait for the next boring meeting.",
    ],
    "label": [1, 1, 0, 1, 1, 1],  # 1 for sarcastic, 0 for not sarcastic
}

# Create a DataFrame
df = pd.DataFrame(data)

# Split the dataset into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Tokenize the text data
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True)
test_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True)

# Convert encodings to PyTorch tensors
train_dataset = torch.utils.data.Dataset()
test_dataset = torch.utils.data.Dataset()

train_dataset = torch.utils.data.TensorDataset(
    torch.tensor(train_encodings['input_ids']),
    torch.tensor(train_encodings['attention_mask']),
    torch.tensor(train_labels.tolist())
)

test_dataset = torch.utils.data.TensorDataset(
    torch.tensor(test_encodings['input_ids']),
    torch.tensor(test_encodings['attention_mask']),
    torch.tensor(test_labels.tolist())
)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Create a Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()

# Evaluate the model
preds = trainer.predict(test_dataset)
pred_labels = torch.argmax(preds.predictions, axis=1)

# Calculate accuracy
accuracy = accuracy_score(test_labels, pred_labels)
print(f'Accuracy: {accuracy:.2f}')
