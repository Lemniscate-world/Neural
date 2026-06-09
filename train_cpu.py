"""train_cpu.py — Train Neural-Agent on CPU (no GPU, no bitsandbytes, no TRL).

Minimal training loop using only transformers + peft on CPU.
Validates the full pipeline: collect -> format -> train -> infer.
"""

import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

# ── Dataset ──────────────────────────────────────────────────────────────────


class TripletDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        t = self.examples[idx]
        instruction = t.get("instruction", "")
        response = json.dumps(t.get("response", {}), ensure_ascii=False)

        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        full = prompt + response

        tokenized = self.tokenizer(
            full,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)

        # Labels: mask prompt tokens with -100
        prompt_len = len(self.tokenizer(prompt)["input_ids"])
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ── Train ────────────────────────────────────────────────────────────────────


def train_on_cpu():
    print("=" * 60)
    print("Neural-Agent CPU Training (pipeline validation)")
    print("=" * 60)

    # 1. Load tiny model
    model_name = "sshleifer/tiny-gpt2"  # 2.8M params, fits anywhere
    print(f"\n[1/5] Loading {model_name}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision="main")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, revision="main", torch_dtype=torch.float32)
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    # 2. Apply LoRA
    print("\n[2/5] Applying LoRA...")
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=["c_attn"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"  Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # 3. Load dataset
    triplet_dir = Path("C:/Users/Utilisateur/Documents/NeuralDBG/artifacts/triplets")
    all_triplets = triplet_dir / "all_triplets.jsonl"

    # Convert triplets to messages format
    formatted_path = triplet_dir / "formatted.jsonl"
    with (
        open(all_triplets, encoding="utf-8") as fin,
        open(formatted_path, "w", encoding="utf-8") as fout,
    ):
        for line in fin:
            t = json.loads(line)
            fout.write(json.dumps(t, ensure_ascii=False, default=str) + "\n")

    print(f"\n[3/5] Loading {formatted_path}...")
    dataset = TripletDataset(str(formatted_path), tokenizer, max_length=1024)
    print(f"  Samples: {len(dataset)}")

    # 4. Train (manual loop, no TRL)
    print("\n[4/5] Training on CPU (5 steps, just validation)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    model.train()

    t0 = time.time()
    losses = []
    for step in range(5):
        batch = dataset[step % len(dataset)]
        input_ids = batch["input_ids"].unsqueeze(0)
        attention_mask = batch["attention_mask"].unsqueeze(0)
        labels = batch["labels"].unsqueeze(0)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())
        print(f"  Step {step + 1}/5: loss={loss.item():.4f}")

    elapsed = time.time() - t0
    print(f"  Training time: {elapsed:.1f}s ({elapsed / 5:.1f}s/step)")

    # 5. Save model
    save_dir = Path("C:/Users/Utilisateur/Documents/NeuralDBG/artifacts/model_final")
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    print(f"\n[5/5] Model saved to {save_dir}")

    # 6. Quick inference test
    print("\n" + "=" * 60)
    print("Inference test")
    print("=" * 60)
    model.eval()
    test_prompt = (
        "### Instruction:\n"
        "Tu es un agent IA de diagnostic ML. Voici les événements NeuralDBG:\n"
        "- gradient_health_transition at fc1.weight step 2: NORMAL -> EXPLODING (confidence 1.0)\n"
        "- nan_detected at loss step 3: 0.0 -> nan (confidence 1.0)\n\n"
        "Quelle est la catégorie de défaillance et quel fix recommandes-tu ?\n\n"
        "### Response:\n"
    )
    input_ids = tokenizer(test_prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=100, do_sample=False)
    response = tokenizer.decode(output[0][input_ids.shape[1] :], skip_special_tokens=True)
    print(f"Input: {test_prompt[:80]}...")
    print(f"Output: {response[:200]}")
    print(f"\nPipeline: collect -> format -> train -> infer : OK")


if __name__ == "__main__":
    train_on_cpu()
