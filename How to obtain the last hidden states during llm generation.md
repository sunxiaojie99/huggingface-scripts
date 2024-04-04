```
from transformers import MistralForCausalLM, AutoTokenizer
import torch
tokenizer = AutoTokenizer.from_pretrained('/home/gomall/models/Mistral-7B-Instruct-v0-1')
model = MistralForCausalLM.from_pretrained('/home/gomall/models/Mistral-7B-Instruct-v0-1').to('cuda')

prompt = "Hey, are you conscious?"
inputs = tokenizer.encode(prompt, return_tensors="pt").to('cuda')

output_sequences = model.generate(input_ids=inputs, max_length=30, output_attentions=True, output_hidden_states=True, return_dict_in_generate=True)

input_hidden = output_sequences.hidden_states[0][-1]  # hidden states of input prompt, shape=[bs, 7, dim]
output_hidden = torch.cat([item[-1].unsqueeze(0) for item in output_sequences.hidden_states[1:]], dim=0)  # hidden states of generate text, shape=[22, bs, 1, dim]
print(input_hidden.shape)
print(output_hidden.shape)

for generated_token_index, hidden_states in enumerate(output_sequences.hidden_states):
    for i, decoder_element in enumerate(hidden_states):
        print(f"Generated token index: {generated_token_index}, decoder element {i} shape: {decoder_element.shape}")
```
