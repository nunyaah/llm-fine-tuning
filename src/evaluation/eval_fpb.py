"""Evaluate a model (base or fine-tuned) on Financial PhraseBank sentences_allagree split."""

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from src.data.prompts import format_inference_prompt, parse_label
from src.evaluation.metrics import compute_metrics, confusion_matrix_df, save_results

FPB_DATASET = "takala/financial_phrasebank"
FPB_CONFIG = "sentences_allagree"
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}


def load_test_set(seed: int = 42, test_size: float = 0.2):
    ds = load_dataset(FPB_DATASET, FPB_CONFIG)["train"]
    split = ds.train_test_split(test_size=test_size, seed=seed)
    return split["test"]


def run_eval(model_id: str, output_path: str | None = None, batch_size: int = 8):
    test_set = load_test_set()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=10,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    preds, labels = [], []
    for row in tqdm(test_set, desc=f"Evaluating {model_id}"):
        prompt = format_inference_prompt(row["sentence"])
        output = pipe(prompt)[0]["generated_text"]
        # Strip the prompt prefix to get only the generated tokens
        generated = output[len(prompt):]
        preds.append(parse_label(generated))
        labels.append(LABEL_MAP[row["label"]])

    metrics = compute_metrics(labels, preds)
    print(f"\n=== Results for {model_id} ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    cm = confusion_matrix_df(labels, preds)
    print("\nConfusion matrix:")
    print(cm)

    if output_path:
        save_results({"model": model_id, **metrics}, output_path)

    return metrics, labels, preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model ID or local path")
    parser.add_argument("--output", default=None, help="Path to save JSON results")
    args = parser.parse_args()
    run_eval(args.model, args.output)


if __name__ == "__main__":
    main()
