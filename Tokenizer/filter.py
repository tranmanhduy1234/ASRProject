import unicodedata
import string

input_path = r"D:\chuyen_nganh\ASRProject\Tokenizer\text2tokenizer.txt"
output_path = r"D:\chuyen_nganh\ASRProject\Tokenizer\text2tokenizer_filtered.txt"

PUNCTUATION = string.punctuation + "“”‘’„"

def clean_text(text):
    text = unicodedata.normalize("NFC", text)
    
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch) not in ("Cc", "Cf")
    )
    
    table = str.maketrans(PUNCTUATION, " " * len(PUNCTUATION))
    text = text.translate(table)
    
    text = " ".join(text.split())
    
    return text

with open(input_path, "r", encoding="utf-8") as fin, \
     open(output_path, "w", encoding="utf-8") as fout:
    for line in fin:
        cleaned = clean_text(line)
        if cleaned: 
            fout.write(cleaned.lower() + "\n")