import torch
from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig
from config.settings import MODEL_ID, COMPUTE_DTYPE, LORA_R, LORA_ALPHA, OUTPUT_DIR, ADAPTER_DIR, GRADIENT_ACC_STEPS, BATCH_SIZE, MAX_SEQ_LENGTH

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
    # 2. CARICAMENTO MODELLO IN 4-BIT (la "Q" di QLoRA)
    # ══════════════════════════════════════════════════════════════════════════════

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",          
        bnb_4bit_compute_dtype=COMPUTE_DTYPE,
        bnb_4bit_use_double_quant=True,     
    )

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

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model)

    print(f"\n [DEBUG] Model uploaded in 4-bit to: {model.device}")

    # ══════════════════════════════════════════════════════════════════════════════
    # 3. LORA CONFIGURATIONS 
    # ══════════════════════════════════════════════════════════════════════════════
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_R,               
        lora_alpha=LORA_ALPHA,      
        target_modules=[    
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Attaching LoRA configurations to the model: adding A and B matrices to every target layers
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"\n [DEBUG] LoRA configurations DONE!")

    # ══════════════════════════════════════════════════════════════════════════════
    # 4. STConfig AND SFTTrainer
    # ══════════════════════════════════════════════════════════════════════════════

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,

        # ── epochs e batches ──────────────────────────────────────────────────────
        num_train_epochs=3,
        per_device_train_batch_size=BATCH_SIZE, 
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=GRADIENT_ACC_STEPS,     
        gradient_checkpointing=True,       
        # ── optimizer ───────────────────────────────────────────────────────
        optim="paged_adamw_8bit",        
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        # ── precision ──────────────────────────────────────────────────────────
        bf16=True,                         
        # ── sequence ────────────────────────────────────────────────────────────
        packing=False,                    
        # ── logs e savings ───────────────────────────────────────────────────
        logging_steps=10,
        eval_strategy="epoch",
        eval_steps=100,
        save_strategy="epoch",
        save_steps=100,
        save_total_limit=2,                
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",                  # "wandb" or "tensorboard"
    )

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