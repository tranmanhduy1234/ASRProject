# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-vi-en")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-vi-en")

import torch

def translate_vi_en(texts, tokenizer, model,
                    max_length=256,
                    num_beams=5,
                    device=None):
    if isinstance(texts, str):
        single_input = True
        texts = [texts]
    else:
        single_input = False

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True
        )

    translations = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True
    )

    return translations[0] if single_input else translations

print("Starting translate...")
index = 1
with open(r"D:\chuyen_nganh\ASRProject\Data\transcription_lintran92.txt", "r", encoding="utf-8") as fin:
  with open(r"D:\chuyen_nganh\ASRProject\Data\transcription_lintran92_en.txt", "w", encoding="utf-8") as fout:
    for line in fin:
        print(index)
        index += 1
        fout.write(translate_vi_en(line, tokenizer, model) + "\n")