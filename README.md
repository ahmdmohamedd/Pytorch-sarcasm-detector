# Sarcasm Detection Model

This repository contains a sarcasm detection model built using PyTorch and the Transformers library. The model utilizes a pre-trained BERT (Bidirectional Encoder Representations from Transformers) model to classify text as sarcastic or not.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Training and Evaluation](#training-and-evaluation)
- [Contributing](#contributing)

## Installation

To run the sarcasm detection model, you need to install the required libraries. You can do this using pip:

```bash
pip install torch torchvision torchaudio
pip install transformers
pip install pandas scikit-learn
pip install h5py
```

If you are using Anaconda, consider creating a new virtual environment to avoid conflicts:

```bash
conda create --name sarcasm_detection python=3.11
conda activate sarcasm_detection
```

## Usage

You can use the model by loading the pre-trained BERT tokenizer and model as follows:

```python
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Example text for sarcasm detection
input_text = "I absolutely love waiting in long lines!"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model(**inputs)

# Get predictions
predictions = torch.argmax(outputs.logits, dim=-1)
print(f"Predicted label: {predictions.item()}")
```

## Training and Evaluation

The script includes both training and evaluation functions. To train and evaluate the model on your own sarcasm dataset, prepare your dataset in a CSV format with two columns: `text` and `label`. The `label` column should contain binary values (0 for non-sarcastic, 1 for sarcastic).

Run the training and evaluation by executing the following command:

```bash
python main.py
```

## Contributing

Contributions are welcome! If you have suggestions for improvements or features, feel free to open an issue or submit a pull request.
