"""Central hyperparameter configuration for QLoRA fine-tuning."""

import torch
from dataclasses import dataclass, field
from typing import List


@dataclass
class LoraConfig:
    r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class BnbConfig:
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: torch.dtype = torch.bfloat16
    bnb_4bit_use_double_quant: bool = True


@dataclass
class TrainingConfig:
    # Model & data
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    output_dir: str = "./qwen25-finsent-qlora"
    hub_model_id: str = ""  # Set to "your-hf-username/qwen25-7b-finsent-qlora"

    # Training
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4   # effective batch = 16
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    max_seq_length: int = 512
    fp16: bool = False
    bf16: bool = True

    # Logging & saving
    logging_steps: int = 25
    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 200
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"

    # W&B
    report_to: str = "wandb"
    run_name: str = "qwen25-finsent-qlora"

    # Dataset split
    val_size: float = 0.05
    seed: int = 42

    # Push to Hub
    push_to_hub: bool = False
