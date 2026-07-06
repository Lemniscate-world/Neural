---
bug_id: PM-004
title: Vanishing gradients via Sigmoid saturation
pytorch_issue: Generic (common failure mode)
pr: N/A
date: 2026-07-06
---

# PM-004: Vanishing gradients via Sigmoid saturation

## Metadata
- **PyTorch Issue**: Generic (common failure mode)
- **PR**: N/A
- **Events Captured**: 42
- **Causal Chains**: 30

## Root Cause
Sigmoid activation saturates at extremes, gradient -> 0

## Symptom
Gradient norm < 1e-6 in deeper layers, no learning

## Causal Chain (NeuralDBG)
```
CausalChain(links=[CausalLink(source_event={'event_type': 'data_anomaly', 'layer_name': 'Linear_0', 'step': 0, 'from_state': 'normal', 'to_state': 'distribution_shift', 'confidence': 0.005936960723417909, 'metadata': {'prev_mean': 0.04892913997173309, 'current_mean': 0.02043491043150425, 'prev_std': 0.9598928093910217, 'current_std': 0.15065297484397888, 'mean_shift_sigma': 0.029684803617089542, 'tensor_cache_path': 'C:/Users/Utilisateur/Documents/NeuralDBG/artifacts/tensor_cache/anomaly_Linear_0_9ec6fd6acae44413b1f45475c47c8eb9.pt', 'memory_spike': False}}, target_event={'event_type': 'gradient_health_transition', 'layer_name': 'Sigmoid_3', 'step': 0, 'from_state': 'NONE', 'to_state': 'vanishing', 'confidence': 1.0, 'metadata': {'current_norm': 2.7939142110966486e-09, 'transition_type': 'baseline', 'memory_spike': False}}, rule='Temporal(0)', confidence=1.0, evidence='data_anomaly at Linear_0 step 0 -> gradient_health_transition at Sigmoid_3 step 0 (gap=0, base_conf=1.04)'), CausalLink(source_event={'event_type': 'gradient_health_transition', 'layer_name': 'Sigmoid_3', 'step': 0, 'from_state': 'NONE', 'to_state': 'vanishing', 'confidence': 1.0, 'metadata': {'current_norm': 2.7939142110966486e-09, 'transition_type': 'baseline', 'memory_spike': False}}, target_event={'event_type': 'gradient_health_transition', 'layer_name': 'Linear_2', 'step': 0, 'from_state': 'NONE', 'to_state': 'vanishing', 'confidence': 1.0, 'metadata': {'current_norm': 6.804737329169086e-10, 'transition_type': 'baseline', 'memory_spike': False}}, rule='Temporal(0)', confidence=0.325, evidence='gradient_health_transition at Sigmoid_3 step 0 -> gradient_health_transition at Linear_2 step 0 (gap=0, base_conf=0.33)'), CausalLink(source_event={'event_type': 'gradient_health_transition', 'layer_name': 'Linear_2', 'step': 0, 'from_state': 'NONE', 'to_state': 'vanishing', 'confidence': 1.0, 'metadata': {'current_norm': 6.804737329169086e-10, 'transition_type': 'baseline', 'memory_spike': False}}, target_event={'event_type': 'gradient_health_transition', 'layer_name': 'Sigmoid_1', 'step': 0, 'from_state': 'NONE', 'to_state': 'vanishing', 'confidence': 1.0, 'metadata': {'current_norm': 3.079314581100334e-10, 'transition_type': 'baseline', 'memory_spike': False}}, rule='Temporal(0)', confidence=0.325, evidence='gradient_health_transition at Linear_2 step 0 -> gradient_health_transition at Sigmoid_1 step 0 (gap=0, base_conf=0.33)'), CausalLink(source_event={'event_type': 'gradient_health_transition', 'layer_name': 'Sigmoid_1', 'step': 0, 'from_state': 'NONE', 'to_state': 'vanishing', 'confidence': 1.0, 'metadata': {'current_norm': 3.079314581100334e-10, 'transition_type': 'baseline', 'memory_spike': False}}, target_event={'event_type': 'gradient_health_transition', 'layer_name': 'Linear_0', 'step': 0, 'from_state': 'NONE', 'to_state': 'vanishing', 'confidence': 1.0, 'metadata': {'current_norm': 7.654413908264601e-11, 'transition_type': 'baseline', 'memory_spike': False}}, rule='Temporal(0)', confidence=0.325, evidence='gradient_health_transition at Sigmoid_1 step 0 -> gradient_health_transition at Linear_0 step 0 (gap=0, base_conf=0.33)'), CausalLink(source_event={'event_type': 'gradient_health_transition', 'layer_name': 'Linear_0', 'step': 0, 'from_state': 'NONE', 'to_state': 'vanishing', 'confidence': 1.0, 'metadata': {'current_norm': 7.654413908264601e-11, 'transition_type': 'baseline', 'memory_spike': False}}, target_event={'event_type': 'gradient_health_transition', 'layer_name': 'Linear_4', 'step': 0, 'from_state': 'NONE', 'to_state': 'healthy', 'confidence': 1.0, 'metadata': {'current_norm': 0.3329148590564728, 'transition_type': 'baseline', 'memory_spike': False}}, rule='Temporal(0)', confidence=0.25, evidence='gradient_health_transition at Linear_0 step 0 -> gradient_health_transition at Linear_4 step 0 (gap=0, base_conf=0.25)'), CausalLink(source_event={'event_type': 'gradient_health_transition', 'layer_name': 'Linear_4', 'step': 0, 'from_state': 'NONE', 'to_state': 'healthy', 'confidence': 1.0, 'metadata': {'current_norm': 0.3329148590564728, 'transition_type': 'baseline', 'memory_spike': False}}, target_event={'event_type': 'gradient_health_transition', 'layer_name': 'Linear_0', 'step': 1, 'from_state': 'vanishing', 'to_state': 'saturated', 'confidence': 1.0, 'metadata': {'prev_norm': 7.654413908264601e-11, 'current_norm': 5.356641486287117e-05, 'transition_type': 'vanishing_to_saturated', 'memory_spike': False}}, rule='Temporal(1)', confidence=0.2, evidence='gradient_health_transition at Linear_4 step 0 -> gradient_health_transition at Linear_0 step 1 (gap=1, base_conf=0.25)')], root_cause='data_anomaly[distribution_shift]', final_symptom='gradient_health_transition[saturated]', confidence=0.4041666666666666, description='data_anomaly(Linear_0)[distribution_shift] -> gradient_health_transition(Sigmoid_3)[vanishing] -> gradient_health_transition(Linear_2)[vanishing] -> gradient_health_transition(Sigmoid_1)[vanishing] -> gradient_health_transition(Linear_0)[vanishing] -> gradient_health_transition(Linear_4)[healthy] -> gradient_health_transition(Linear_0)[saturated]')
```

## Fix
Replace Sigmoid with ReLU/LeakyReLU, use BatchNorm

## Detection Metrics
{
  "events": 42,
  "chains": 30,
  "vanishing_events": 4
}

---
*Generated by NeuralDBG Post-Mortem Suite v1.0*
