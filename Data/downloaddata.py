from datasets import load_dataset, Audio #type: ignore
import soundfile as sf #type: ignore
import io
import os
import json
from tqdm import tqdm

OUTPUT_FOLDER = r"E:\ASR_DATASET\Audio"
MANIFEST_PATH = r"E:\ASR_DATASET\Doc\vivoid\train_manifest.jsonl" #
global_index = 910000000 #

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
print("Đang kết nối tới Hugging Face...")

ds = load_dataset("capleaf/viVoice", split='train', streaming=True) #
ds = ds.cast_column("audio", Audio(decode=False))

print(f"Bắt đầu tải và xử lý. Manifest sẽ được lưu tại: {MANIFEST_PATH}")
with open(MANIFEST_PATH, "w", encoding="utf-8", errors='ignore') as f_out:
    for i, sample in tqdm(enumerate(ds), desc="Processing"):
        try:
            audio_bytes = sample['audio']['bytes']
            text = sample.get('transcription') or sample.get('text') or ""
            file_id = f"sample_{i + global_index}"
            filename = f"{file_id}.wav"
            save_path = os.path.join(OUTPUT_FOLDER, filename)
            with io.BytesIO(audio_bytes) as f:
                data, samplerate = sf.read(f)
            sf.write(save_path, data, samplerate)
            duration = len(data) / samplerate

            entry = {
                "file": filename,
                "text": text,
                "duration": round(duration, 4),
                "original_sample_rate": samplerate
            }
            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            if i % 100 == 0:
                f_out.flush()
        except Exception as e:
            print(f"\n[Lỗi] Mẫu {i}: {e}")
            continue
print("\nHoàn tất! Kiểm tra folder Audio và file jsonl.")