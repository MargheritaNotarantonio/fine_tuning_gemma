import torch
from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig
from config.settings import MODEL_ID, OUTPUT_DIR, ADAPTER_DIR


def load_tuned_model():

  """Upload the base model and LoRA weights"""

  model_id = ADAPTER_DIR
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  # Pad Token Configuration
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = "right"
  
  # 4-bit configurations also during inference
  bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.bfloat16
  )

  # Upload base model
  base_model = AutoModelForCausalLM.from_pretrained(
      model_id,
      # quantization_config=bnb_config,
      torch_dtype=torch.float16,
      device_map="auto",
      trust_remote_code=True
  )

  # Applying saved LoRA adapter to the model 
  model = PeftModel.from_pretrained(base_model, ADAPTER_DIR, is_trainable=False)
  model.eval() 

  print("[DEBUG] Uploading model and adapter succeded!")
  
  return model, tokenizer