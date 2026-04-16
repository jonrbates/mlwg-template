import json
import os
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from tickets.data import load_data
from tickets.freezing import FreezingCallback

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)

data = load_data()
train_data = data["train"]
eval_data = data["test"].select(range(1024))


def tokenize(example):
    return tokenizer(example["content"], truncation=True, padding="max_length", max_length=128)


train_dataset = train_data.map(tokenize, batched=True).rename_column("label", "labels")
eval_dataset = eval_data.map(tokenize, batched=True).rename_column("label", "labels")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": (predictions == labels).mean()}


run_name = "train-layer-norm-5"

layer_norm_module_names = []
for i in range(6):
    layer_norm_module_names.append(f"distilbert.transformer.layer.{i}.sa_layer_norm")
    layer_norm_module_names.append(f"distilbert.transformer.layer.{i}.output_layer_norm")

freezing_schedule = [
    (0, ["pre_classifier", "classifier"] + layer_norm_module_names),
]

training_args = TrainingArguments(
    output_dir=f"./runs/logs/{run_name}",
    eval_strategy="steps",
    learning_rate=1e-3,
    warmup_ratio=0.05,
    eval_steps=500,
    save_steps=500,
    save_total_limit=2,
    logging_steps=50,
    num_train_epochs=10,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    gradient_accumulation_steps=4,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    report_to="tensorboard",
    logging_dir=f"./runs/logs/{run_name}",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    callbacks=[
        FreezingCallback(freezing_schedule),
        EarlyStoppingCallback(early_stopping_patience=20),
    ],
)

# Save run metadata before training
meta_path = os.path.join(training_args.logging_dir, "run_metadata.json")
os.makedirs(training_args.logging_dir, exist_ok=True)
total = sum(p.numel() for p in model.parameters())

# Final trainable = all modules in the schedule

visited = set()
final_trainable = 0
for _, mods in freezing_schedule:
    for mod_name in mods:
        for p in model.get_submodule(mod_name).parameters():
            if id(p) in visited:
                continue
            final_trainable += p.numel()
            visited.add(id(p))


metadata = {
    "run_name": run_name,
    "trainable_params": final_trainable,
    "total_params": total,
    "learning_rate": training_args.learning_rate,
    "per_device_train_batch_size": training_args.per_device_train_batch_size,
}
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)

# Train and automatically log metrics
# tensorboard --logdir ./runs/logs
trainer.train()
