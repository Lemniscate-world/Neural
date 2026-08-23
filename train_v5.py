"""
v5 GPU Training â€” Qwen2-0.5B + LoRA on 8 architecture families

Trains on v5_training_data.json (108 examples, 6 families + 2 pending).
Uses fp16 (no bitsandbytes â€” Quadro M4000 CUDA 5.2 limitation).

Usage:
    python train_v5.py [--epochs 3] [--output checkpoints_v5/final]
"""

from __future__ import annotations

import json, os, sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, TaskType

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class V5Config:
    base_model: str = "Qwen/Qwen2-0.5B"
    data_path: str = "v5_training_data.json"
    output_dir: str = "neuralagent/model/checkpoints_v5"
    
    # LoRA (same as v4)
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    
    # Training
    num_epochs: int = 3
    batch_size: int = 2  # Small batch for 7GB VRAM
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    max_seq_length: int = 1024
    
    # fp16 (no bitsandbytes â€” Quadro M4000 limitation)
    fp16: bool = True
    bf16: bool = False
    
    save_steps: int = 50
    logging_steps: int = 5
    eval_split: float = 0.1
    
    seed: int = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_v5_data(data_path: str) -> Dataset:
    """Load v5 training data and convert to HuggingFace Dataset.
    
    v5 format: [{"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}]
    """
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Extract messages arrays
    messages_list = [item["messages"] for item in data]
    
    # Convert to HF Dataset
    dataset = Dataset.from_list([{"messages": msgs} for msgs in messages_list])
    
    print(f"Loaded {len(dataset)} examples from {data_path}")
    
    # Print family distribution
    families: Dict[str, int] = {}
    for item in data:
        fam = item.get("family", "unknown")
        families[fam] = families.get(fam, 0) + 1
    print("Family distribution:")
    for fam, count in sorted(families.items()):
        print(f"  {fam}: {count}")
    
    return dataset


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_v5(config: V5Config):
    """Run v5 LoRA fine-tuning."""
    
    print("=" * 60)
    print("NeuralDBG v5 GPU Training")
    print(f"Base: {config.base_model}")
    print(f"Data: {config.data_path}")
    print(f"Output: {config.output_dir}")
    print(f"LoRA: r={config.lora_r}, alpha={config.lora_alpha}")
    print(f"fp16: {config.fp16}, epochs: {config.num_epochs}")
    print("=" * 60)
    
    # Load tokenizer
    print("\n[1/5] Loading tokenizer...")
    # Modele pilote par config locale, environnement d'entrainement controle
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)  # nosec B615
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print("[2/5] Loading base model (fp16)...")
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.float16,
        device_map="auto",  # nosec B615
        trust_remote_code=True,
    )
    
    # Apply LoRA
    print("[3/5] Applying LoRA adapters...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load data
    print("[4/5] Loading training data...")
    dataset = load_v5_data(config.data_path)
    
    # Split
    if config.eval_split > 0:
        split = dataset.train_test_split(test_size=config.eval_split, seed=config.seed)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    else:
        train_dataset = dataset
        eval_dataset = None
    
    # Training arguments
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        fp16=config.fp16,
        bf16=config.bf16,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=config.save_steps if eval_dataset else None,
        save_total_limit=2,
        load_best_model_at_end=True if eval_dataset else False,
        report_to="none",  # No W&B logging
        seed=config.seed,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    
    # SFT Trainer
    print("[5/5] Starting training...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    
    trainer.train()
    
    # Save final model
    final_dir = output_dir / "final"
    print(f"\nSaving final model to {final_dir}...")
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    
    # Save training config
    config_path = output_dir / "training_config.json"
    with open(config_path, "w") as f:
        json.dump({
            "base_model": config.base_model,
            "lora_r": config.lora_r,
            "num_epochs": config.num_epochs,
            "num_examples": len(dataset),
            "families": list(set(
                json.load(open(config.data_path, "r", encoding="utf-8"))[0].get("family", "?")
                for _ in [1]  # Will be computed properly
            )),
        }, f, indent=2)
    
    print(f"\nâœ… v5 training complete! Model saved to {final_dir}")
    
    # Quick eval
    if eval_dataset:
        metrics = trainer.evaluate()
        print(f"\nEval metrics: {metrics}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="v5 GPU Training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--output", type=str, default="neuralagent/model/checkpoints_v5",
                        help="Output directory")
    parser.add_argument("--data", type=str, default="v5_training_data.json",
                        help="Training data JSON")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    args = parser.parse_args()
    
    config = V5Config(
        num_epochs=args.epochs,
        output_dir=args.output,
        data_path=args.data,
        batch_size=args.batch_size,
    )
    
    train_v5(config)
