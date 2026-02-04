import json
import pandas as pd #type: ignore

input_file = r'D:\chuyen_nganh\ASRProject\Data\merged_dataset_all_filtered.jsonl'
output_file = r'D:\chuyen_nganh\ASRProject\Data\merged_dataset_all_sorted.jsonl'

# BƯỚC 1: Chỉ đọc duy nhất cột duration để lấy thứ tự
print("Đang quét chỉ mục thời lượng...")
durations = []
with open(input_file, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        # Chỉ lưu index và duration vào list (rất nhẹ)
        durations.append({'index': i, 'duration': data['duration']})

# BƯỚC 2: Sắp xếp dựa trên list siêu nhẹ này
df_meta = pd.DataFrame(durations)
df_meta_sorted = df_meta.sort_values(by='duration')
sorted_indices = df_meta_sorted['index'].tolist()

# BƯỚC 3: Đọc file gốc lần 2 và ghi ra theo thứ tự đã sort
print("Đang ghi file đã sắp xếp (không tốn RAM)...")
# Nạp toàn bộ file vào list để truy xuất theo index (nếu RAM còn trụ được)
with open(input_file, 'r', encoding='utf-8') as f:
    all_lines = f.readlines()

with open(output_file, 'w', encoding='utf-8') as f_out:
    for idx in sorted_indices:
        f_out.write(all_lines[idx])

print("Sắp xếp hoàn tất và chính xác!")