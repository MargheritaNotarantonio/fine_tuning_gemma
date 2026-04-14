import torch
from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig
from config.settings import MODEL_ID, COMPUTE_DTYPE, ADAPTER_DIR, bnb_config, lora_config, training_args

def create_conversation(sample):
    return {
        "messages": [
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": str(sample["answer"])} 
        ]
    }

def train ():
        
    # ══════════════════════════════════════════════════════════════════════════════
    # 1. ACCESS TO HUGGINGFACE AND DATASET DOWNLOAD
    # ══════════════════════════════════════════════════════════════════════════════
    login()
    print("[DEBUG] Login succeded!")

    dataset = load_dataset("microsoft/wiki_qa", split = "train").shuffle(seed = 42).select(range(1000))
    # To ensure a conversation setting
    dataset = dataset.map(create_conversation, remove_columns=dataset.column_names)

    # Splitting into training and test, setting a seed for reproducibility
    dataset = dataset.train_test_split(test_size=0.2, seed=42, shuffle = False)

    print(f"[DEBUG] Train examples : {len(dataset['train'])}")
    print(f"[DEBUG] Eval  examples : {len(dataset['test'])}")

    # ══════════════════════════════════════════════════════════════════════════════
    # 2. UPLOADING MODELLO IN 4-BIT 
    # ══════════════════════════════════════════════════════════════════════════════

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0},                  # Automatic allocation on GPU/CPU
        trust_remote_code=True,
        attn_implementation="eager",        
        # To minimize RAM usage
        low_cpu_mem_usage=True,
        torch_dtype=COMPUTE_DTYPE 
    )

    # No caching
    model.config.use_cache = False

    # Tokenizer from HuggingFace repository, only the path is needed
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print(f"\n [DEBUG] Model uploaded in 4-bit to: {model.device}")

    # ══════════════════════════════════════════════════════════════════════════════
    # 3. LORA CONFIGURATIONS 
    # ══════════════════════════════════════════════════════════════════════════════
    model = prepare_model_for_kbit_training(model)

    # Attaching LoRA configurations to the model: adding A and B matrices to every target layers
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"\n [DEBUG] LoRA configurations DONE!")

    # ══════════════════════════════════════════════════════════════════════════════
    # 4. SFTTrainer
    # ══════════════════════════════════════════════════════════════════════════════

    trainer = SFTTrainer(
        model=model,                       
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"], 
        processing_class=tokenizer
    )

    print("\n [DEBUG] Starting training the model...\n")
    trainer.train() 
    trainer.model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print("\n [DEBUG] Training succeded!\n")

if __name__ == "__main__":
    train()