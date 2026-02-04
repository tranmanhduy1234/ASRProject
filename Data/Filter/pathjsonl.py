import os
from tqdm import tqdm

root_path = r"E:\ASR_DATASET\Doc"

relative_paths_jsonl = [
    r'infore1_25hours\train_manifest.jsonl',
    r'infore2_audiobooks\train_manifest.jsonl',
    r'linhtran92\test_manifest.jsonl',
    r'linhtran92\train_manifest.jsonl',
    r'linhtran92\validation_manifest.jsonl',
    r'LSVSC\test_manifest.jsonl',
    r'LSVSC\train_manifest.jsonl',
    r'LSVSC\validation_manifest.jsonl',
    r'nhuP\train_manifest.jsonl',
    r'pnnbao-ump_VieNeu-TTS-140h\train_manifest.jsonl',
    r'thivux\test_manifest.jsonl',
    r'thivux\train_manifest.jsonl',
    r'thivux\validation_manifest.jsonl',
    r'viMD\test_manifest.jsonl',
    r'viMD\train_manifest.jsonl',
    r'viMD\validation_manifest.jsonl',
    r'vivoid\train_manifest.jsonl',
    r'vlsp2020_vinai_100h\train_manifest.jsonl',
]

def mergefile():
    output_file = "merged_dataset_all.jsonl"
    print(f"Đang gom file vào: {output_file}...")
    count = 0

    with open(output_file, 'w', encoding='utf-8') as f_out:
        
        for rel_path in tqdm(relative_paths_jsonl):
            full_path = os.path.join(root_path, rel_path)
            
            if not os.path.exists(full_path):
                print(f"[SKIP - Không tìm thấy]: {full_path}")
                continue

            # Đọc và chép từng dòng sang file mới
            with open(full_path, 'r', encoding='utf-8') as f_in:
                for line in f_in:
                    clean_line = line.strip()
                    if clean_line:
                        f_out.write(clean_line + '\n')
                        count += 1
    print("------------------------------------------------")
    print("ĐÃ XONG!")
    print(f"Tổng số dòng đã gom: {count}")
    
def pathjsonslist():
    from pathlib import Path

    root_dir = Path(r"E:\ASR_DATASET\Doc")  # ví dụ: "data"

    json_files = [p.relative_to(root_dir) for p in root_dir.rglob("*.jsonl")]

    for path in json_files:
        print("'" + str(path) + "'")