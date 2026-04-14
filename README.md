# FINE TUNING WITH GEMMA (270M) 
With this project I wanted to share what I've learned about fine tuning a model on local GPU.
The model I chose is '[google/gemma-3-270m-it](https://huggingface.co/google/gemma-3-270m-it)' and the dataset is '[microsoft/wiki_qa](https://huggingface.co/datasets/microsoft/wiki_qa)' from HuggingFace.

🚀 In this project the key factors are: 
1) Fine Tuning with LoRA (4-bit) to minimize the VRAM usage
2) Support for HuggingFace datasets
3) Inference script optimised for local GPUs.

To set up the project properly I decided to use the [uv](https://docs.astral.sh/uv/) library: 

```python 
project_fine_tuning/
├── config/                 # Configurations
│   └── settings.py         # MODEL_ID, Hyperparameters, bnb_config
├── final_adapter/          # Trained Adapter LoRA addestrati (weights)
│   ├── adapter_model.safetensors
│   └── tokenizer.json
├── training.py             # Fine-tuning script on Colab/GPU
├── inference.py            # Inference Script 
├── load_tuned_model.py     # Model loading script 
└── README.md               # Documentation
```

🛠️ Installation

1. Clone of repository

```python 
git clone https://github.com/MargheritaNotarantonio/fine_tuning_gemma
cd fine_tuning_gemma
```

2. Installation of dependecies (it is strongly suggested to use a virtual environment)

pip install torch transformers peft bitsandbytes accelerate datasets

⚙️ How to use it

1. Fine-Tuning
To train the model (GPU NVIDIA with at least 8GB is needed or you can use Google Colab):

```python 
uv run python training.py
```

The weights will be saved in final_adapter/. folder

2. Inference

```python 
uv run inference.py
```

💡 Technical Notes

1. Quantisation: The model is loaded in 4-bit (nf4) format to enable fine-tuning on consumer hardware.

2. Chat Template: The official Gemma 3 template is used to correctly handle conversation turns between <|user|> and <|assistant|>.

3. Local Deployment: If you are running the script locally without an NVIDIA GPU, ensure you remove the bitsandbytes configuration and load the model in float32.

👨‍💻 Autore
M. Notarantonio 