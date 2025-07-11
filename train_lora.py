import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

# --- Setup ---
# Correctly identify the device
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

base_model = "tiiuae/falcon-7b"

# --- Model Loading ---
# **THE FIX IS HERE:**
# We are now loading the model in float32 instead of float16.
# While float16 is great for inference on MPS, the backward pass during training
# can fail because not all gradient operations are supported for float16 on MPS.
# Loading in float32 is more stable for training.
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float32,  # Use float32 for training on MPS
    trust_remote_code=True,
    device_map=device, # Use device_map to place the model on the correct device
)

# --- Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(base_model)
# A pad token is required for training. Using the EOS token is a common strategy.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- PEFT / LoRA Configuration ---

# We REMOVE the call to `prepare_model_for_kbit_training(model)`.
# This function is specifically for bitsandbytes quantization, which caused the
# initial meta tensor error on your MPS device.

# Configure LoRA.
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=[
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Verify the setup by printing the number of trainable parameters.
# This should be a small fraction of the total parameters.
model.print_trainable_parameters()