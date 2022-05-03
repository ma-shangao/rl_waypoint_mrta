# GNNClustering_add_attention

# Policy Structure
- **the transformer encoder**: one shot encoding
- **the modified transformer decoder**: decode step by step:
  - **state** : at each node(timestep), the state changes are same as the paper 'Attention, Learn to Solve Routing Problems!'
  - **action** : at each node(timestep), we concate the output query of the decoder and the attention scores to decode our goups dividing probabilities.
  - **rewards** : just the same as your idea.

# How to run
```
python end2end_attention_policy_dev.py
```

# TODO lists
- please help me run this process and save the models
- please help me evaluate it
