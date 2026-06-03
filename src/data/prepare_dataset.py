"""Load FinGPT sentiment train dataset, filter to standard labels, and format with ChatML."""

from datasets import load_dataset, DatasetDict
from src.data.prompts import SYSTEM_PROMPT, VALID_LABELS, format_training_example

FINGPT_DATASET = "FinGPT/fingpt-sentiment-train"
LABEL_NORMALISATION = {
    "mildly positive": "positive",
    "mildly negative": "negative",
    "strong positive": "positive",
    "strong negative": "negative",
    "moderately positive": "positive",
    "moderately negative": "negative",
}


def _normalise_label(label: str) -> str:
    label = label.strip().lower()
    return LABEL_NORMALISATION.get(label, label)


def _format_row(row: dict) -> dict:
    instruction = row.get("instruction", SYSTEM_PROMPT)
    input_text = row.get("input", "")
    output = _normalise_label(row.get("output", ""))
    return {
        "text": format_training_example(instruction, input_text, output),
        "label": output,
    }


def load_and_prepare(seed: int = 42, val_size: float = 0.05) -> DatasetDict:
    ds = load_dataset(FINGPT_DATASET, split="train")

    # Normalise labels and drop rows with unrecognised labels
    ds = ds.map(_format_row, remove_columns=ds.column_names)
    ds = ds.filter(lambda row: row["label"] in VALID_LABELS)

    split = ds.train_test_split(test_size=val_size, seed=seed)
    return DatasetDict({"train": split["train"], "validation": split["test"]})


if __name__ == "__main__":
    dataset = load_and_prepare()
    print(dataset)
    print(dataset["train"][0]["text"])
