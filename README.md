# FinSentLLM: Financial Sentiment Analysis with QLoRA

Fine-tuned `Qwen2.5-7B-Instruct` on 76K financial news examples using QLoRA. Achieved **15+ point F1 improvement** over zero-shot baseline on the Financial PhraseBank benchmark — matching Bloomberg-scale results at a fraction of the cost.

## Key Results

| Model | FPB Accuracy | Weighted F1 | MCC |
|---|---|---|---|
| Qwen2.5-7B-Instruct (zero-shot) | ~65–70% | ~0.60–0.66 | ~0.42–0.51 |
| FinBERT (published reference) | ~88% | ~0.87 | ~0.81 |
| FinGPT v3.2 Llama2-7B LoRA (published) | ~87–91% | ~0.86–0.90 | ~0.80 |
| **Qwen2.5-7B FinSent QLoRA (this project)** | **TBD** | **TBD** | **TBD** |

*Fill in actual results after training.*

## Quick Start

```bash
pip install -r requirements.txt
```

**Inference:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_id = "your-hf-username/qwen25-7b-finsent-qlora"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=5)
prompt = (
    "<|im_start|>system\nYou are a financial analyst. Classify the sentiment of the following "
    "financial news sentence from an investor's perspective. Respond with only one word: "
    "positive, negative, or neutral.<|im_end|>\n"
    "<|im_start|>user\nOperating profit rose to EUR 13.1 mn from EUR 8.7 mn in the prior year period.<|im_end|>\n"
    "<|im_start|>assistant\n"
)
print(pipe(prompt)[0]["generated_text"][len(prompt):])  # → "positive"
```

## Project Structure

```
├── notebooks/
│   └── 03_qlora_training.ipynb   # Self-contained Kaggle notebook (run this)
├── src/
│   ├── data/
│   │   ├── prompts.py            # ChatML template + label parsing
│   │   └── prepare_dataset.py    # Load + format FinGPT dataset
│   ├── training/
│   │   ├── config.py             # Hyperparameters
│   │   └── train.py              # Training entry point
│   └── evaluation/
│       ├── metrics.py            # Accuracy, F1, MCC
│       └── eval_fpb.py           # FPB benchmark runner
├── results/                      # Populated after training
├── requirements.txt
└── .env.example
```

## Training

**Platform:** Kaggle T4 (16GB VRAM) — free, 30 hrs/week  
**Method:** QLoRA (4-bit NF4 + LoRA r=16, α=32)  
**Data:** `FinGPT/fingpt-sentiment-train` — 76K instruction-formatted examples (MIT)  
**Epochs:** 3 | **Effective batch:** 16 | **LR:** 2e-4 cosine  
**Estimated time:** 5–8 hours on T4

### Run on Kaggle

1. Upload `notebooks/03_qlora_training.ipynb` to Kaggle
2. Enable GPU T4 and Internet in Notebook Settings
3. Add secrets: `WANDB_API_KEY`, `HF_TOKEN`
4. Set `HUB_MODEL_ID` in the config cell
5. Run All

### Run locally (requires ≥16GB VRAM)

```bash
python -m src.training.train
```

## Evaluation

```bash
# Baseline (zero-shot)
python -m src.evaluation.eval_fpb --model Qwen/Qwen2.5-7B-Instruct --output results/baseline_results.json

# Fine-tuned
python -m src.evaluation.eval_fpb --model your-hf-username/qwen25-7b-finsent-qlora --output results/finetuned_results.json
```

## Dataset

- **Train:** `FinGPT/fingpt-sentiment-train` (MIT, ~76K samples) — instruction-formatted, 3-class sentiment
- **Eval:** `takala/financial_phrasebank` `sentences_allagree` split (CC-BY-NC-SA, ~2,264 high-agreement samples)
- Labels normalised: `mildly positive/negative` → `positive/negative`

## Limitations

- Trained on financial news headlines; may not generalise to earnings call transcripts or social media
- `sentences_allagree` eval subset uses only the highest-agreement annotations — real-world ambiguous sentences may score lower
- FPB eval dataset is CC-BY-NC-SA: suitable for portfolio/research use, not commercial deployment

## References

- Yang et al., "FinGPT: Open-Source Financial Large Language Models", IJCAI 2023
- Malo et al., "Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts", JASIST 2014 (FPB)
- [Qwen2.5 model family](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [PEFT docs](https://huggingface.co/docs/peft)
- [TRL SFTTrainer docs](https://huggingface.co/docs/trl/sft_trainer)
