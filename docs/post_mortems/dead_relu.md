---
bug_id: PM-005
title: Dead neurons via zero initialization + negative bias
pytorch_issue: Generic (common failure mode)
pr: N/A
date: 2026-07-05
---

# PM-005: Dead neurons via zero initialization + negative bias

## Metadata
- **PyTorch Issue**: Generic (common failure mode)
- **PR**: N/A
- **Events Captured**: 50
- **Causal Chains**: 30

## Root Cause
Zero weights + negative bias → all ReLU outputs = 0

## Symptom
Zero gradients or constant output, no learning

## Causal Chain (NeuralDBG)
```
CausalChain(links=[CausalLink(source_event={'event_type': 'data_anomaly', 'layer_name': 'Linear_0', 'step': 0, 'from_state': 'normal', 'to_state': 'distribution_shift', 'confidence': 1.0, 'metadata': {'prev_mean': 0.042101841419935226, 'current_mean': -10.0, 'prev_std': 0.9418650269508362, 'current_std': 0.0, 'mean_shift_sigma': 10.661933030818561, 'tensor_cache_path': 'C:/Users/Utilisateur/Documents/NeuralDBG/artifacts/tensor_cache/anomaly_Linear_0_e5e4b88deecd4f19be3a86a44912eace.pt', 'memory_spike': False}}, target_event={'event_type': 'gradient_health_transition', 'layer_name': 'ReLU_3', 'step': 0, 'from_state': 'NONE', 'to_state': 'vanishing', 'confidence': 1.0, 'metadata': {'current_norm': 0.0, 'transition_type': 'baseline', 'memory_spike': False}}, rule='Temporal(0)', confidence=1.0, evidence='data_anomaly at Linear_0 step 0 -> gradient_health_transition at ReLU_3 step 0 (gap=0, base_conf=1.04)'), CausalLink(source_event={'event_type': 'gradient_health_transition', 'layer_name': 'ReLU_3', 'step': 0, 'from_state': 'NONE', 'to_state': 'vanishing', 'confidence': 1.0, 'metadata': {'current_norm': 0.0, 'transition_type': 'baseline', 'memory_spike': False}}, target_event={'event_type': 'activation_regime_shift', 'layer_name': 'Linear_0', 'step': 0, 'from_state': 'NONE', 'to_state': 'saturated', 'confidence': 1.0, 'metadata': {'mean': -10.0, 'std': 0.0, 'min': -10.0, 'max': -10.0, 'sparsity': 0.0, 'dead_ratio': 0.0, 'norm': 226.274169921875, 'saturation_ratio': 1.0, 'memory_spike': False}}, rule='Temporal(0)', confidence=0.52, evidence='gradient_health_transition at ReLU_3 step 0 -> activation_regime_shift at Linear_0 step 0 (gap=0, base_conf=0.52)'), CausalLink(source_event={'event_type': 'activation_regime_shift', 'layer_name': 'Linear_0', 'step': 0, 'from_state': 'NONE', 'to_state': 'saturated', 'confidence': 1.0, 'metadata': {'mean': -10.0, 'std': 0.0, 'min': -10.0, 'max': -10.0, 'sparsity': 0.0, 'dead_ratio': 0.0, 'norm': 226.274169921875, 'saturation_ratio': 1.0, 'memory_spike': False}}, target_event={'event_type': 'gradient_health_transition', 'layer_name': 'Linear_2', 'step': 0, 'from_state': 'NONE', 'to_state': 'vanishing', 'confidence': 1.0, 'metadata': {'current_norm': 0.0, 'transition_type': 'baseline', 'memory_spike': False}}, rule='Temporal(0)', confidence=0.78, evidence='activation_regime_shift at Linear_0 step 0 -> gradient_health_transition at Linear_2 step 0 (gap=0, base_conf=0.78)'), CausalLink(source_event={'event_type': 'gradient_health_transition', 'layer_name': 'Linear_2', 'step': 0, 'from_state': 'NONE', 'to_state': 'vanishing', 'confidence': 1.0, 'metadata': {'current_norm': 0.0, 'transition_type': 'baseline', 'memory_spike': False}}, target_event={'event_type': 'activation_regime_shift', 'layer_name': 'ReLU_1', 'step': 0, 'from_state': 'NONE', 'to_state': 'dead', 'confidence': 1.0, 'metadata': {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'sparsity': 1.0, 'dead_ratio': 1.0, 'norm': 0.0, 'saturation_ratio': 0.0, 'memory_spike': False}}, rule='Temporal(0)', confidence=0.52, evidence='gradient_health_transition at Linear_2 step 0 -> activation_regime_shift at ReLU_1 step 0 (gap=0, base_conf=0.52)'), CausalLink(source_event={'event_type': 'activation_regime_shift', 'layer_name': 'ReLU_1', 'step': 0, 'from_state': 'NONE', 'to_state': 'dead', 'confidence': 1.0, 'metadata': {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'sparsity': 1.0, 'dead_ratio': 1.0, 'norm': 0.0, 'saturation_ratio': 0.0, 'memory_spike': False}}, target_event={'event_type': 'gradient_health_transition', 'layer_name': 'Linear_0', 'step': 0, 'from_state': 'NONE', 'to_state': 'vanishing', 'confidence': 1.0, 'metadata': {'current_norm': 0.0, 'transition_type': 'baseline', 'memory_spike': False}}, rule='Temporal(0)', confidence=0.78, evidence='activation_regime_shift at ReLU_1 step 0 -> gradient_health_transition at Linear_0 step 0 (gap=0, base_conf=0.78)'), CausalLink(source_event={'event_type': 'gradient_health_transition', 'layer_name': 'Linear_0', 'step': 0, 'from_state': 'NONE', 'to_state': 'vanishing', 'confidence': 1.0, 'metadata': {'current_norm': 0.0, 'transition_type': 'baseline', 'memory_spike': False}}, target_event={'event_type': 'activation_regime_shift', 'layer_name': 'Linear_2', 'step': 0, 'from_state': 'NONE', 'to_state': 'saturated', 'confidence': 1.0, 'metadata': {'mean': -10.0, 'std': 0.0, 'min': -10.0, 'max': -10.0, 'sparsity': 0.0, 'dead_ratio': 0.0, 'norm': 160.0, 'saturation_ratio': 1.0, 'memory_spike': False}}, rule='Temporal(0)', confidence=0.52, evidence='gradient_health_transition at Linear_0 step 0 -> activation_regime_shift at Linear_2 step 0 (gap=0, base_conf=0.52)')], root_cause='data_anomaly[distribution_shift]', final_symptom='activation_regime_shift[saturated]', confidence=0.6866666666666666, description='data_anomaly(Linear_0)[distribution_shift] -> gradient_health_transition(ReLU_3)[vanishing] -> activation_regime_shift(Linear_0)[saturated] -> gradient_health_transition(Linear_2)[vanishing] -> activation_regime_shift(ReLU_1)[dead] -> gradient_health_transition(Linear_0)[vanishing] -> activation_regime_shift(Linear_2)[saturated]')
```

## Fix
Use nn.init.kaiming_uniform_ or Xavier initialization

## Detection Metrics
{
  "events": 50,
  "chains": 30
}

---
*Generated by NeuralDBG Post-Mortem Suite v1.0*
