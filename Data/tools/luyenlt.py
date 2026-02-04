import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-vi-en")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-vi-en")

model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Hàm dịch 1 câu
def translate_vi_en(text, max_length=256):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            num_beams=8
        )

    return tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )

# Đường dẫn file
input_path = r"D:\chuyen_nganh\ASRProject\Data\thivux\transcription_thivux_validation.txt"
output_path = r"D:\chuyen_nganh\ASRProject\Data\thivux\transcription_thivux_validation_EN.txt"

# Dịch từng dòng
with open(input_path, "r", encoding="utf-8") as fin, \
     open(output_path, "w", encoding="utf-8") as fout:
    index = 1
    for line in fin:
        line = line.strip()
        if not line:
            fout.write("\n")
            continue
        print(index)
        index += 1
        en = translate_vi_en(line)
        fout.write(en + "\n")