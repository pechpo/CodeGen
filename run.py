import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import json
from safetensors.torch import load_file
from tqdm import tqdm

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,4,5,6'

# Paths
#checkpoint = "./saved_models/code_generation/final_checkpoint/model.safetensors"
checkpoint = "./saved_models/code_generation/final_checkpoint"
tokenizer_checkpoint = "./codet5p-220m"
test_data_path = "./CodeT5/CodeT5p/dataset/test.json"
output_data_path = "./CodeT5/CodeT5p/dataset/test_output.json"

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, torch_dtype=torch.float16, trust_remote_code=True).to(device)
#model = AutoModelForSeq2SeqLM.from_pretrained(tokenizer_checkpoint, torch_dtype=torch.float16, trust_remote_code=True)

## Load model weights from safetensors file
#model_weights = load_file(checkpoint)
#model.load_state_dict(model_weights)
#model.to(device)

# Load test data
with open(test_data_path, 'r') as f:
    test_data = [json.loads(line) for line in f]

# Generate predictions
for entry in tqdm(test_data):
    encoding = tokenizer(entry['nl'], return_tensors="pt", max_length=512).to(device)
    encoding['decoder_input_ids'] = torch.tensor([[0]]).to(device)
    outputs = model.generate(**encoding, max_length=1000)
    entry['code'] = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Save predictions
with open(output_data_path, 'w') as f:
    for entry in test_data:
        json.dump(entry, f)
        f.write('\n')

print("Predictions saved to", output_data_path)
