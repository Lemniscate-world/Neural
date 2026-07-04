"""Generate RNN-enriched training data from combinatorial validation results.

Uses the fixed NeuralDBG (RNN-aware hooks) to capture hidden state events,
gate saturation, and BPTT gradient health from LSTM/GRU architectures.
Appends to the existing training dataset.

Usage: python generate_rnn_training_data.py
Output: rnn_enriched_train.jsonl (ready to merge with existing train.jsonl)
"""

import sys, json, random
sys.path.insert(0, r"C:\Users\Utilisateur\Documents\NeuralDBG")
sys.path.insert(0, r"C:\Users\Utilisateur\Documents\Neural-Agent")

import torch, torch.nn as nn
from neuraldbg import NeuralDbg
from validate_combinatorial import *

torch.manual_seed(42)
random.seed(42)

SYSTEM_PROMPT = (
    "Tu es un agent IA expert en diagnostic ML. Tu reçois des événements capturés "
    "par NeuralDBG lors de l'entraînement d'un réseau de neurones PyTorch. "
    "Tu dois : (1) identifier la catégorie de défaillance, (2) expliquer la cause racine, "
    "(3) recommander un fix concret avec des paramètres spécifiques. "
    "Réponds toujours en JSON structuré."
)

FIX_TEMPLATES = {
    "exploding_gradients": {
        "category": "exploding_gradients",
        "root_cause": "Exploding gradients detected — gradient norms exceed safe threshold.",
        "fix": {
            "type": "hyperparameter",
            "action": "decrease_lr_and_add_gradient_clipping",
            "lr_multiplier": 0.1,
            "clip_grad_norm": 1.0,
            "description": "Decrease LR by 10x, add gradient clipping at norm=1.0."
        }
    },
    "vanishing_gradients": {
        "category": "vanishing_gradients",
        "root_cause": "Vanishing gradients — gradient norms collapsed below 1e-6 during BPTT.",
        "fix": {
            "type": "hyperparameter",
            "action": "increase_lr_and_use_skip_connections",
            "lr_multiplier": 2.0,
            "add_skip_connections": True,
            "use_layernorm": True,
            "description": "Increase LR by 2x, add skip connections and LayerNorm after RNN layers."
        }
    },
    "rnn_hidden_saturation": {
        "category": "rnn_hidden_saturation",
        "root_cause": "RNN hidden state saturation — >70% of hidden units saturated (sigmoid/tanh).",
        "fix": {
            "type": "hyperparameter",
            "action": "decrease_lr_and_reduce_forget_bias",
            "lr_multiplier": 0.5,
            "forget_gate_bias": -1.0,
            "use_gradient_clipping": True,
            "description": "Decrease LR by 2x, reduce forget gate bias to -1.0, add gradient clipping."
        }
    },
    "dead_neurons": {
        "category": "dead_neurons",
        "root_cause": "Dead neurons — >90% of activations permanently zero.",
        "fix": {
            "type": "hyperparameter",
            "action": "decrease_lr_and_use_leaky_relu",
            "lr_multiplier": 0.5,
            "suggested_activation": "LeakyReLU",
            "description": "Decrease LR by 2x, swap to LeakyReLU to revive dead neurons."
        }
    },
    "nan_data": {
        "category": "nan_data",
        "root_cause": "NaN detected in input data — data pipeline corruption.",
        "fix": {
            "type": "data_pipeline",
            "action": "filter_nan_batches_and_normalize",
            "nan_filter": True,
            "normalize": "batch_norm",
            "description": "Filter NaN batches, apply batch normalization to inputs."
        }
    },
    "divergence": {
        "category": "divergence",
        "root_cause": "Training divergence — loss increased by >100x in one step.",
        "fix": {
            "type": "hyperparameter",
            "action": "decrease_lr_and_add_warmup",
            "lr_multiplier": 0.01,
            "warmup_steps": 100,
            "description": "Decrease LR by 100x, add 100-step warmup."
        }
    },
}


def bug_to_category(bug_name: str) -> str:
    mapping = {
        "exploding": "exploding_gradients",
        "vanishing": "vanishing_gradients",
        "zero_init": "dead_neurons",
        "nan_data": "nan_data",
        "dead_bias": "dead_neurons",
        "divergence": "divergence",
    }
    return mapping.get(bug_name, "unknown")


def events_to_text(events, max_events=25):
    """Convert NeuralDBG events to text for the prompt."""
    lines = []
    for e in events[:max_events]:
        et = e.event_type.value if hasattr(e, 'event_type') else str(e.get("event_type", "?"))
        layer = e.layer_name if hasattr(e, 'layer_name') else e.get("layer_name", "?")
        step = e.step if hasattr(e, 'step') else e.get("step", "?")
        from_s = e.from_state if hasattr(e, 'from_state') else e.get("from_state", "?")
        to_s = e.to_state if hasattr(e, 'to_state') else e.get("to_state", "?")
        conf = e.confidence if hasattr(e, 'confidence') else e.get("confidence", 1.0)
        lines.append(f"- {et} at {layer} step {step}: {from_s} -> {to_s} (confidence {conf:.3f})")
    return "\n".join(lines)


def generate_example(cfg: ArchConfig, bug_name: str, bug_fn, events):
    """Generate one training example from a buggy run."""
    cat = bug_to_category(bug_name)
    template = FIX_TEMPLATES.get(cat, FIX_TEMPLATES["divergence"])
    
    event_text = events_to_text(events)
    
    user_text = (
        f"Tu es un agent IA de diagnostic ML. Voici les événements NeuralDBG "
        f"d'un entraînement PyTorch défaillant (architecture: {cfg.name}, "
        f"famille: {cfg.family}). Analyse et propose un fix.\n\n"
        f"Events:\n{event_text}\n\n"
        f"Quelle est la catégorie de défaillance et quel fix recommandes-tu ?"
    )
    
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": json.dumps(template, ensure_ascii=False, indent=2)},
        ]
    }


def main():
    print("Generating RNN-enriched training data...")
    
    # Focus on RNN + Hybrid configs (the ones that were broken before the fix)
    rnn_cfgs = rnn_configs(15)
    hybrid_cfgs = hybrid_configs(10)
    all_cfgs = rnn_cfgs + hybrid_cfgs
    
    examples = []
    skipped = 0
    
    for cfg in all_cfgs:
        try:
            model = cfg.make_model()
            data_fn = cfg.make_data
            
            # Run with each bug type
            for bug_name, bug_fn in BUGS:
                try:
                    model2 = cfg.make_model()
                    ev, _, _ = train_with_dbg(model2, data_fn, steps=8, bug=bug_fn)
                    
                    # Convert SemanticEvent objects to dicts if needed
                    event_list = []
                    for e in ev:
                        if hasattr(e, 'event_type'):
                            event_list.append(e)
                        elif isinstance(e, dict):
                            event_list.append(e)
                    
                    if len(event_list) >= 5:  # Need enough events to be meaningful
                        ex = generate_example(cfg, bug_name, bug_fn, event_list)
                        examples.append(ex)
                    else:
                        skipped += 1
                except Exception:
                    skipped += 1
        except Exception as e:
            skipped += 1
    
    # Write output
    out_path = "rnn_enriched_train.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    
    print(f"  Generated {len(examples)} RNN-enriched training examples")
    print(f"  Skipped {skipped} (not enough events or errors)")
    print(f"  Saved to: {out_path}")
    
    # Summary by category
    from collections import Counter
    cats = Counter()
    for ex in examples:
        resp = json.loads(ex["messages"][2]["content"])
        cats[resp["category"]] += 1
    print(f"\n  Category distribution:")
    for cat, count in cats.most_common():
        print(f"    {cat}: {count}")


if __name__ == "__main__":
    main()
