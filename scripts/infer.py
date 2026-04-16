"""Run interactive inference with a saved checkpoint.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# checkpoint = "./runs/checkpoint-24500"
checkpoint = "jonrbates/tickets-sentiment"

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
model.eval()

labels = ["negative", "positive"]

print(f"Model loaded from {checkpoint}. Enter text (Ctrl+C to quit):\n")
try:
    while True:
        text = input("> ").strip()
        if not text:
            continue
        inputs = tokenizer([text], truncation=True, padding=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        probability_score = probs[1]
        print(f"{probability_score:.1%}\n")
except (KeyboardInterrupt, EOFError):
    print()
