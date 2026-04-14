import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from config.settings import MODEL_ID, ADAPTER_DIR, OUTPUT_DIR, bnb_config
from load_tuned_model import load_tuned_model

def inference(model, tokenizer, prompt):

  messages = [{"role": "user", "content": prompt}]

  input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

  inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(model.device)

  inputs.pop("token_type_ids", None)

  with torch.no_grad():
      outputs = model.generate(
          **inputs,
          max_new_tokens=200,
          do_sample=True,
          temperature=0.7,
          top_p=0.9,
          pad_token_id=tokenizer.pad_token_id,
          eos_token_id=tokenizer.eos_token_id
      )

  # Decodifica solo i nuovi token (esclude il prompt)
  new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
  return tokenizer.decode(new_tokens, skip_special_tokens=True)

if __name__ == "__main__":  

  model, tokenizer = load_tuned_model()

  print("\n--- Modello pronto! Scrivi 'esci' per terminare ---")
    
  while True:
      user_input = input("\nTu: ")
      if user_input.lower() in ["esci", "exit", "quit"]:
          break
          
      risposta = inference(model, tokenizer, user_input)
      print(f"\nGemma (Fine-tuned): {risposta}")
