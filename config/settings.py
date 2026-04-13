import torch

# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURAZIONE GENERALE
# ══════════════════════════════════════════════════════════════════════════════

# Modello e Path 
MODEL_ID = "google/gemma-3-270m-it" 
OUTPUT_DIR = "./gemma-finetuned-wiki" # Output for checkpoints
ADAPTER_DIR = "./final_adapter"

# Hardware & VRAM Optimization
MAX_SEQ_LENGTH = 512  # Ridotto per 7GB VRAM
BATCH_SIZE = 4
GRADIENT_ACC_STEPS = 2
COMPUTE_DTYPE = torch.bfloat16 # o float16 se la GPU è pre-RTX 3000

# LoRA Params
LORA_R = 16
LORA_ALPHA = 32

learning_rate = 5e-5
