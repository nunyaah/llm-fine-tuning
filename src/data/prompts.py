SYSTEM_PROMPT = (
    "You are a financial analyst. Classify the sentiment of the following financial news "
    "sentence from an investor's perspective. Respond with only one word: positive, negative, or neutral."
)

CHATML_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{input}<|im_end|>\n"
    "<|im_start|>assistant\n{output}<|im_end|>"
)

CHATML_INFERENCE_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{input}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

VALID_LABELS = {"positive", "negative", "neutral"}


def format_training_example(instruction: str, input_text: str, output: str) -> str:
    return CHATML_TEMPLATE.format(
        system=instruction or SYSTEM_PROMPT,
        input=input_text,
        output=output,
    )


def format_inference_prompt(input_text: str) -> str:
    return CHATML_INFERENCE_TEMPLATE.format(system=SYSTEM_PROMPT, input=input_text)


def parse_label(generated_text: str) -> str:
    """Extract the first valid sentiment label from model output."""
    text = generated_text.strip().lower()
    for label in VALID_LABELS:
        if label in text:
            return label
    return "neutral"
