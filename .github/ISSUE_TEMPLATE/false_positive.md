---
name: "🎯 False Positive Report"
about: "Report a case where NeuralDBG detected a failure that wasn't there (or missed one)"
title: "[FALSE POSITIVE] "
labels: ["validation", "causal-engine"]
assignees: []
---

## Description
<!-- Describe the scenario where NeuralDBG gave a wrong diagnosis. -->

## Scenario Details
- **Architecture**: <!-- e.g. MLP, Transformer, CNN -->
- **Failure Type**: <!-- e.g. Vanishing Gradients, Exploding Gradients, NaN -->
- **Expected Diagnosis**: <!-- What should NeuralDBG have said? -->
- **Actual Diagnosis**: <!-- What did NeuralDBG actually say? -->

## Code Snippet
<!-- If possible, provide a minimal code snippet to reproduce the issue. -->
```python
# Your code here
```

## Logs / Output
<!-- Paste the output from NeuralDBG here. -->

## Additional Context
<!-- Any other details that might help us improve the causal engine. -->
