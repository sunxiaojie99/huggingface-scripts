[official tutorial](https://huggingface.co/docs/transformers/v4.39.3/en/llm_tutorial#generate-text)

1. whether padding side affects the encoding of the model
```
from transformers import MistralForCausalLM, AutoTokenizer
import torch

encoder = MistralForCausalLM.from_pretrained('models/Mistral-7B-Instruct-v0-1').to('cuda')

#  whether padding side affects the encoding of the model

# right
tokenizer = AutoTokenizer.from_pretrained('models/Mistral-7B-Instruct-v0-1', padding_side="right")
tokenizer.pad_token = tokenizer.unk_token  # Most LLMs don't have a pad token by default
model_inputs1 = tokenizer(["1, 2, 3", "A, B, C, D, E, F, G"], padding=True, return_tensors="pt").to("cuda")
out1 = encoder.model(**model_inputs1, return_dict=True)

# left
tokenizer = AutoTokenizer.from_pretrained("models/Mistral-7B-Instruct-v0-1", padding_side="left")
tokenizer.pad_token = tokenizer.unk_token  # Most LLMs don't have a pad token by default
model_inputs2 = tokenizer(["1, 2, 3", "A, B, C, D, E, F, G"], padding=True, return_tensors="pt").to("cuda")
out2 = encoder.model(**model_inputs2, return_dict=True)

"""
padding side affects the encoding of the model

(Pdb) out2.last_hidden_state
tensor([[[ 0.2642,  7.1616,  2.6754,  ..., -0.5677, -0.3455,  1.5414],
         [ 0.2642,  7.1616,  2.6754,  ..., -0.5677, -0.3455,  1.5414],
         [ 0.2642,  7.1616,  2.6754,  ..., -0.5677, -0.3455,  1.5414],
         ...,
         [-5.5222,  1.5238,  1.7341,  ..., -0.2664, -3.5843,  3.6744],
         [-5.7893, -1.0254, -1.3950,  ..., -2.7380,  3.0445, -0.8982],
         [ 2.1663,  2.1185,  1.3178,  ...,  2.0029, -5.2990, -0.7156]],

        [[-2.0509,  1.9392, -2.1350,  ..., -0.1247, -2.2761,  2.9441],
         [-1.2903,  4.0531,  1.8107,  ..., -3.3903,  0.7747,  1.5240],
         [ 0.5332,  1.8658,  4.1448,  ...,  0.2268,  1.9287,  0.9519],
         ...,
         [-0.7953,  2.7091, -0.2635,  ...,  8.2592,  3.6784, -5.5153],
         [-4.0939,  3.4491,  1.9479,  ...,  0.6998,  3.9296, -0.4598],
         [-3.1506, -0.3979, -1.0231,  ...,  6.3552,  2.1078, -4.6987]]],
       device='cuda:0', grad_fn=<MulBackward0>)
(Pdb) out1.last_hidden_state
tensor([[[-2.0509,  1.9392, -2.1350,  ..., -0.1247, -2.2761,  2.9441],
         [-1.1939,  5.3754,  0.5580,  ..., -3.9021,  1.7466,  0.5687],
         [ 2.3387,  2.8613,  4.7814,  ..., -3.5847,  0.3889,  4.3802],
         ...,
         [ 0.8954,  3.6660,  1.8329,  ...,  0.1817, -0.6310,  0.3653],
         [ 0.9021,  3.6641,  1.8736,  ...,  0.1842, -0.5148,  0.4081],
         [ 0.8725,  3.7211,  1.9186,  ...,  0.1527, -0.4980,  0.4063]],

        [[-2.0509,  1.9392, -2.1350,  ..., -0.1247, -2.2761,  2.9441],
         [-1.2903,  4.0531,  1.8107,  ..., -3.3903,  0.7747,  1.5240],
         [ 0.5332,  1.8658,  4.1448,  ...,  0.2268,  1.9287,  0.9519],
         ...,
         [-0.7953,  2.7091, -0.2635,  ...,  8.2592,  3.6784, -5.5153],
         [-4.0939,  3.4491,  1.9479,  ...,  0.6998,  3.9296, -0.4598],
         [-3.1506, -0.3979, -1.0231,  ...,  6.3552,  2.1078, -4.6987]]],
       device='cuda:0', grad_fn=<MulBackward0>)
"""

```
