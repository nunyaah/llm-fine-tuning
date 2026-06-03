"""QLoRA fine-tuning entry point for Qwen2.5-7B-Instruct on FinGPT sentiment data."""

import os
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

from src.data.prepare_dataset import load_and_prepare
from src.training.config import TrainingConfig, LoraConfig as LoraCfg, BnbConfig


def build_bnb_config(cfg: BnbConfig) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=cfg.bnb_4bit_compute_dtype,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
    )


def build_lora_config(cfg: LoraCfg) -> LoraConfig:
    return LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.target_modules,
        lora_dropout=cfg.lora_dropout,
        bias=cfg.bias,
        task_type=cfg.task_type,
    )


def train(cfg: TrainingConfig | None = None):
    cfg = cfg or TrainingConfig()

    wandb.init(project="finsent-qlora", name=cfg.run_name)

    # ---------- tokenizer ----------
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ---------- model ----------
    bnb_config = build_bnb_config(BnbConfig())
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, build_lora_config(LoraCfg()))
    model.print_trainable_parameters()

    # ---------- data ----------
    dataset = load_and_prepare(seed=cfg.seed, val_size=cfg.val_size)

    # ---------- trainer ----------
    training_args = SFTConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        logging_steps=cfg.logging_steps,
        eval_strategy=cfg.eval_strategy,
        eval_steps=cfg.eval_steps,
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        report_to=cfg.report_to,
        run_name=cfg.run_name,
        push_to_hub=cfg.push_to_hub,
        hub_model_id=cfg.hub_model_id or None,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=training_args,
        max_seq_length=cfg.max_seq_length,
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    if cfg.push_to_hub and cfg.hub_model_id:
        trainer.push_to_hub()

    wandb.finish()
    return trainer


if __name__ == "__main__":
    train()
