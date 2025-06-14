from huggingface_hub import login
import os
os.environ["HF_HUB_OFFLINE"] = "True"
login() 
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-2-7b-hf"  # or 13b, 70b if you have resources
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)