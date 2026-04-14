import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig

# Model and Paths 
MODEL_ID = "google/gemma-3-270m-it" 
OUTPUT_DIR = "./gemma-finetuned-wiki" # Output for checkpoints
ADAPTER_DIR = "./final_adapter"

# Hardware & VRAM Optimization
MAX_SEQ_LENGTH = 512  
BATCH_SIZE = 4
GRADIENT_ACC_STEPS = 2
COMPUTE_DTYPE = torch.bfloat16 

# Training settings
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          
    bnb_4bit_compute_dtype=COMPUTE_DTYPE,
    bnb_4bit_use_double_quant=True    
)

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

# LoRA Parameters Configuration
LORA_R = 16
LORA_ALPHA = 32

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
