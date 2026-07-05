---
bug_id: PM-007
title: Training divergence via extreme learning rate (LR=500)
pytorch_issue: Generic (common failure mode)
pr: N/A
date: 2026-07-05
---

# PM-007: Training divergence via extreme learning rate (LR=500)

## Metadata
- **PyTorch Issue**: Generic (common failure mode)
- **PR**: N/A
- **Events Captured**: 29
- **Causal Chains**: 30

## Root Cause
LR=500 causes loss to diverge to inf in <8 steps

## Symptom
Loss spikes → inf, optimizer instability

## Causal Chain (NeuralDBG)
```
CausalChain(links=[CausalLink(source_event={'event_type': 'data_anomaly', 'layer_name': 'ReLU_1', 'step': 2, 'from_state': 'normal', 'to_state': 'distribution_shift', 'confidence': 1.0, 'metadata': {'prev_mean': 20.44028663635254, 'current_mean': -10047.1513671875, 'prev_std': 35.114566802978516, 'current_std': 29731.5625, 'mean_shift_sigma': 286.70698716892304, 'tensor_cache_path': 'C:/Users/Utilisateur/Documents/NeuralDBG/artifacts/tensor_cache/anomaly_ReLU_1_3b8acca4301b4d2ebc86e701edba554d.pt', 'memory_spike': False}}, target_event={'event_type': 'data_anomaly', 'layer_name': 'Linear_2', 'step': 2, 'from_state': 'normal', 'to_state': 'distribution_shift', 'confidence': 0.09368624302072927, 'metadata': {'prev_mean': 3666.443603515625, 'current_mean': -800.0, 'prev_std': 9534.8974609375, 'current_std': 852743424.0, 'mean_shift_sigma': 0.4684312151036463, 'tensor_cache_path': 'C:/Users/Utilisateur/Documents/NeuralDBG/artifacts/tensor_cache/anomaly_Linear_2_a111fc233ad94af2aaa3298ea7986a53.pt', 'memory_spike': False}}, rule='Temporal(0)', confidence=0.65, evidence='data_anomaly at ReLU_1 step 2 -> data_anomaly at Linear_2 step 2 (gap=0, base_conf=0.65)'), CausalLink(source_event={'event_type': 'data_anomaly', 'layer_name': 'Linear_2', 'step': 2, 'from_state': 'normal', 'to_state': 'distribution_shift', 'confidence': 0.09368624302072927, 'metadata': {'prev_mean': 3666.443603515625, 'current_mean': -800.0, 'prev_std': 9534.8974609375, 'current_std': 852743424.0, 'mean_shift_sigma': 0.4684312151036463, 'tensor_cache_path': 'C:/Users/Utilisateur/Documents/NeuralDBG/artifacts/tensor_cache/anomaly_Linear_2_a111fc233ad94af2aaa3298ea7986a53.pt', 'memory_spike': False}}, target_event={'event_type': 'data_anomaly', 'layer_name': 'ReLU_1', 'step': 2, 'from_state': 'distribution_shift', 'to_state': 'normal', 'confidence': 1.0, 'metadata': {'memory_spike': False}}, rule='Temporal(0)', confidence=0.5, evidence='data_anomaly at Linear_2 step 2 -> data_anomaly at ReLU_1 step 2 (gap=0, base_conf=0.50)'), CausalLink(source_event={'event_type': 'data_anomaly', 'layer_name': 'ReLU_1', 'step': 2, 'from_state': 'distribution_shift', 'to_state': 'normal', 'confidence': 1.0, 'metadata': {'memory_spike': False}}, target_event={'event_type': 'gradient_health_transition', 'layer_name': 'Linear_0', 'step': 2, 'from_state': 'healthy', 'to_state': 'exploding', 'confidence': 1.0, 'metadata': {'prev_norm': 55.34474182128906, 'current_norm': 12448.91796875, 'transition_type': 'healthy_to_exploding', 'memory_spike': False}}, rule='Temporal(0)', confidence=0.8, evidence='data_anomaly at ReLU_1 step 2 -> gradient_health_transition at Linear_0 step 2 (gap=0, base_conf=0.80)'), CausalLink(source_event={'event_type': 'gradient_health_transition', 'layer_name': 'Linear_0', 'step': 2, 'from_state': 'healthy', 'to_state': 'exploding', 'confidence': 1.0, 'metadata': {'prev_norm': 55.34474182128906, 'current_norm': 12448.91796875, 'transition_type': 'healthy_to_exploding', 'memory_spike': False}}, target_event={'event_type': 'gradient_health_transition', 'layer_name': 'ReLU_1', 'step': 2, 'from_state': 'healthy', 'to_state': 'exploding', 'confidence': 1.0, 'metadata': {'prev_norm': 76.01898193359375, 'current_norm': 18584.3984375, 'transition_type': 'healthy_to_exploding', 'memory_spike': False}}, rule='Temporal(0)', confidence=0.325, evidence='gradient_health_transition at Linear_0 step 2 -> gradient_health_transition at ReLU_1 step 2 (gap=0, base_conf=0.33)'), CausalLink(source_event={'event_type': 'gradient_health_transition', 'layer_name': 'ReLU_1', 'step': 2, 'from_state': 'healthy', 'to_state': 'exploding', 'confidence': 1.0, 'metadata': {'prev_norm': 76.01898193359375, 'current_norm': 18584.3984375, 'transition_type': 'healthy_to_exploding', 'memory_spike': False}}, target_event={'event_type': 'optimizer_instability', 'layer_name': 'optimizer', 'step': 5, 'from_state': 'stable', 'to_state': 'loss_spike', 'confidence': 0.85, 'metadata': {'current_loss': 6.512147611695186e+17}}, rule='Temporal(3)', confidence=0.364, evidence='gradient_health_transition at ReLU_1 step 2 -> optimizer_instability at optimizer step 5 (gap=3, base_conf=0.91)')], root_cause='data_anomaly[distribution_shift]', final_symptom='optimizer_instability[loss_spike]', confidence=0.5278, description='data_anomaly(ReLU_1)[distribution_shift] -> data_anomaly(Linear_2)[distribution_shift] -> data_anomaly(ReLU_1)[normal] -> gradient_health_transition(Linear_0)[exploding] -> gradient_health_transition(ReLU_1)[exploding] -> optimizer_instability(optimizer)[loss_spike]')
```

## Fix
Reduce LR to ≤0.01, add gradient clipping, use learning rate scheduler

## Detection Metrics
{
  "events": 29,
  "chains": 30
}

---
*Generated by NeuralDBG Post-Mortem Suite v1.0*
