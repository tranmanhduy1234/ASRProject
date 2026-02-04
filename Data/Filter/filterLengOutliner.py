import json

file_path_input = r"D:\chuyen_nganh\ASRProject\Data\merged_dataset_all.jsonl"
file_path_output = r"D:\chuyen_nganh\ASRProject\Data\merged_dataset_all_filtered.jsonl"

with open(file_path_input, 'r', encoding='utf-8') as f_in:
    with open(file_path_output, 'w', encoding='utf-8', errors='ignore') as f_out:
        for i, line in enumerate(f_in):
            data = json.loads(line)
            duration = float(data.get('duration'))
            if duration < 1 or duration > 20:
                continue
            entry = {
                "file": data.get('file'),
                "text": data.get('text'),
                "duration": data.get('duration'),
                "original_sample_rate": data.get('original_sample_rate')
            }
            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            if i % 100 == 0:
                f_out.flush()