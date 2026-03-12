import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tickets.data import load_data

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)


class SequenceAveragingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_module = bert_model.distilbert.embeddings.word_embeddings
        self.a_linear_module = nn.Linear(768, 2)

    def forward(self, x, attention_mask):
        # TODO


model = SequenceAveragingModel()

data = load_data()
train_data = list(data["train"].take(32))


def evaluate_model(model, data):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for item in data:
            inputs = tokenizer(item["content"], return_tensors="pt", truncation=True)
            output = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
            predicted_label = torch.argmax(output, dim=1).item()
            if predicted_label == item["label"]:
                correct += 1
            total += 1
    print(f"Accuracy: {correct / total:.2%}")


evaluate_model(model, train_data)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for epoch in range(5):
    model.train()
    for item in train_data:
        model.zero_grad()
        # TODO
        inputs =
        label =
        output =
        loss =
        # TODO: gradient descent

    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

evaluate_model(model, train_data)
