import asyncio
from googletrans import Translator

INPUT_FILE = r"D:\chuyen_nganh\ASRProject\Data\nhuP\transcription_nhuP_vietspeech.txt"   # file tiếng Việt
OUTPUT_FILE = r"D:\chuyen_nganh\ASRProject\Data\nhuP\transcription_nhuP_vietspeech_en.txt"  # file tiếng Anh

async def translate_file():
    async with Translator() as translator:
        with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
             open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
            index = 1
            for line in fin:
                print(index, line)
                index += 1
                text = line.strip()
                if not text:
                    fout.write("\n")
                    continue

                try:
                    result = await translator.translate(text, src="vi", dest="en")
                    fout.write(result.text + "\n")
                except Exception as e:
                    # ghi dòng lỗi để không mất alignment
                    fout.write("[TRANSLATION_ERROR]\n")
                    print(f"Lỗi với câu: {text} | {e}")

asyncio.run(translate_file())