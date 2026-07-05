"""
v5 Training Data Exporter

Generates French-format training examples for 8 architecture families:
MLP, CNN, RNN, TF, Hybrid, GNN, MoE, Diffusion.

Each example = {
    "prompt": French prompt with event summary,
    "completion": JSON with category, diagnosis, fix
}

Matches the training format used by v4 (Qwen2-0.5B + LoRA).

Usage:
    python export_v5_data.py --families all --output v5_training_data.json
"""

import sys, json, random, argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import our test infrastructure
from validate_combinatorial import (
    BUGS, bug_exploding, bug_vanishing, bug_zero, bug_nan, bug_dead, bug_divergence,
    ALL_ARCHS, ArchConfig, make_model, train_arch, n_problematic,
)
from validate_blackswans import (
    BLACKSWAN_BUILDERS, gnn_configs, moe_configs, diffusion_configs,
    train_blackswan,
    GNNModel, SparseMoE, MoEModel, DiffusionUNet, GCNLayer,
)

torch.manual_seed(42)
random.seed(42)

# ---------------------------------------------------------------------------
# French prompt templates (matching v4 training format)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "Tu es un agent IA expert en diagnostic ML. Tu recois des evenements "
    "captures par NeuralDBG lors de l'entrainement d'un reseau de neurones PyTorch. "
    "Tu dois : (1) identifier la categorie de defaillance, (2) expliquer la cause racine, "
    "(3) proposer un correctif precis."
)

CATEGORY_MAP = {
    "exploding": "explosion_de_gradient",
    "vanishing": "disparition_de_gradient",
    "nan_data": "contamination_NaN",
    "zero_init": "initialisation_zero",
    "dead_bias": "neurones_morts",
    "divergence": "divergence_entrainement",
}

FAMILY_MAP = {
    "MLP": "perceptron_multicouche",
    "CNN": "reseau_convolutif",
    "RNN": "reseau_recurrent",
    "TF": "transformeur",
    "Hybrid": "architecture_hybride",
    "GNN": "reseau_de_graphes",
    "MoE": "melange_d_experts",
    "Diffusion": "modele_de_diffusion",
}


def build_prompt(events: List[Dict], family: str, arch_name: str) -> str:
    """Build a French prompt matching the v4 training format."""
    event_lines = []
    for e in events[:20]:  # Truncate to 20 events
        et = e.get("event_type", e.get("type", "?"))
        ln = e.get("layer_name", "?")
        conf = e.get("confidence", 0.0)
        event_lines.append(
            f"- [{et}] couche={ln}, confiance={conf:.2f}"
        )

    events_text = "\n".join(event_lines) if event_lines else "Aucun evenement detecte"

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Architecture: {FAMILY_MAP.get(family, family)} ({arch_name})\n"
        f"Nombre d'evenements: {len(events)}\n\n"
        f"Evenements:\n{events_text}\n\n"
        f"Question: Quel est le probleme et comment le corriger ?"
    )


def build_completion(bug_name: str, events: List[Dict], family: str) -> str:
    """Build the expected JSON completion."""
    category = CATEGORY_MAP.get(bug_name, bug_name)

    fix_suggestions = {
        "explosion_de_gradient": "Reduire le taux d'apprentissage (lr=0.001) et ajouter du gradient clipping (max_norm=1.0)",
        "disparition_de_gradient": "Utiliser BatchNorm apres chaque couche et remplacer Sigmoid par ReLU. Verifier l'initialisation Xavier.",
        "contamination_NaN": "Ajouter une verification torch.isnan(x) avant le forward. Nettoyer les donnees d'entree.",
        "initialisation_zero": "Utiliser nn.init.xavier_uniform_ ou nn.init.kaiming_normal_ au lieu de zeros.",
        "neurones_morts": "Remplacer les ReLU par LeakyReLU(0.01) ou ELU. Reduire le learning rate.",
        "divergence_entrainement": "Ajouter gradient clipping (max_norm=1.0). Reduire le learning rate. Verifier la normalisation des donnees.",
    }

    fix = fix_suggestions.get(category, "Analyser les evenements et corriger la cause racine.")

    problem_events = [e for e in events
                      if e.get("event_type", "") in
                      ("data_anomaly", "nan_detected", "silent_corruption",
                       "optimizer_instability", "gradient_health_transition")]

    has_nan = any("nan" in str(e.get("event_type", "")).lower() for e in events)
    has_explosion = any("explosion" in str(e.get("event_type", "")).lower() or
                        e.get("to_state") == "exploding" for e in events)
    severity = "critique" if has_nan else ("elevee" if has_explosion else "moderee")

    completion = {
        "categorie": category,
        "diagnostic": (
            f"Defaillance de type '{category}' detectee dans une architecture "
            f"{FAMILY_MAP.get(family, family)}. {len(problem_events)} evenements "
            f"problematiques sur {len(events)} evenements totaux."
        ),
        "severite": severity,
        "correctif": fix,
        "confiance": 0.85 + random.uniform(-0.05, 0.10),
    }

    return json.dumps(completion, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Example generation
# ---------------------------------------------------------------------------

def generate_examples_for_family(
    family: str,
    configs: List[ArchConfig],
    bugs_to_use: List[Tuple[str, Any]],
    steps: int = 8,
) -> List[Dict]:
    """Generate training examples for one architecture family."""
    examples = []
    for cfg in configs:
        for bug_name, bug_fn in bugs_to_use:
            try:
                if family in ("GNN", "MoE", "Diffusion"):
                    events, _, _ = train_blackswan(cfg, steps=steps, bug=bug_fn)
                else:
                    model = make_model(cfg)
                    events, _, _ = train_arch(model, cfg, data_fn=None,
                                              steps=steps, bug=bug_fn)

                # Only include if we have meaningful events
                if len(events) >= 2:
                    prompt = build_prompt(events, family, cfg.name)
                    completion = build_completion(bug_name, events, family)

                    examples.append({
                        "family": family,
                        "arch": cfg.name,
                        "bug": bug_name,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": completion},
                        ],
                        "num_events": len(events),
                    })
            except Exception as e:
                print(f"  SKIP {family}/{cfg.name}/{bug_name}: {e}")

        print(f"  {cfg.name}: {sum(1 for ex in examples if ex['arch']==cfg.name)} exemples")

    return examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="v5 Training Data Exporter")
    parser.add_argument("--families", default="all",
                        help="Comma-separated families or 'all'")
    parser.add_argument("--output", default="v5_training_data.json",
                        help="Output JSON file")
    parser.add_argument("--steps", type=int, default=8,
                        help="Training steps per example")
    parser.add_argument("--per-family", type=int, default=6,
                        help="Max configs per family")
    args = parser.parse_args()

    if args.families == "all":
        families_to_export = ["MLP", "CNN", "RNN", "TF", "Hybrid", "GNN", "MoE", "Diffusion"]
    else:
        families_to_export = [f.strip() for f in args.families.split(",")]

    bugs_to_use = BUGS  # All 6 bugs

    all_examples: List[Dict] = []

    print("=" * 60)
    print("v5 Training Data Exporter")
    print(f"Families: {families_to_export}")
    print(f"Bugs: {[b[0] for b in bugs_to_use]}")
    print("=" * 60)

    for family in families_to_export:
        print(f"\n--- {family} ---")

        if family in ("GNN", "MoE", "Diffusion"):
            # Black-swan families
            if family == "GNN":
                configs = gnn_configs(args.per_family)
            elif family == "MoE":
                configs = moe_configs(args.per_family)
            else:
                configs = diffusion_configs(args.per_family)
        else:
            # Standard families from combinatorial tester
            configs = [c for c in ALL_ARCHS if c.family == family][:args.per_family]

        if not configs:
            print(f"  No configs found for {family}")
            continue

        examples = generate_examples_for_family(family, configs, bugs_to_use, steps=args.steps)
        all_examples.extend(examples)
        print(f"  Total: {len(examples)} exemples")

    # Save
    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_examples, f, ensure_ascii=False, indent=2)

    # Statistics
    by_family: Dict[str, int] = {}
    by_bug: Dict[str, int] = {}
    for ex in all_examples:
        by_family[ex["family"]] = by_family.get(ex["family"], 0) + 1
        by_bug[ex["bug"]] = by_bug.get(ex["bug"], 0) + 1

    print(f"\n{'='*60}")
    print(f"EXPORTED: {len(all_examples)} exemples -> {output_path}")
    print(f"\nBy family:")
    for fam, count in sorted(by_family.items()):
        print(f"  {fam}: {count}")
    print(f"\nBy bug:")
    for bug, count in sorted(by_bug.items()):
        print(f"  {bug}: {count}")

    # Estimate training time
    total_examples = len(all_examples)
    print(f"\nEstimated v5 training: ~{total_examples * 0.5:.0f}s per epoch "
          f"(Qwen2-0.5B LoRA r=8, Quadro M4000)")


if __name__ == "__main__":
    main()
