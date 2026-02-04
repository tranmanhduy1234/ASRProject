import json

# Cấu hình đường dẫn
input_sorted_file = r'D:\chuyen_nganh\ASRProject\Data\merged_dataset_all_sorted.jsonl'
output_jsonl = r'D:\chuyen_nganh\ASRProject\Data\combined_metadata.jsonl'

def combine_jsonl_metadata(target_duration=15.0, silence_duration=0.5):
    combined_results = []
    
    # Khởi tạo batch hiện tại
    current_batch = {
        "audio_paths": [],
        "transcripts": [],
        "total_duration": 0.0
    }
    
    batch_count = 0

    print("Đang bắt đầu gom nhóm dữ liệu...")

    with open(input_sorted_file, 'r', encoding='utf-8') as f_in, \
         open(output_jsonl, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            line = line.strip()
            if not line:
                continue
                
            item = json.loads(line)
            # Giả sử file jsonl có các trường: 'file' (hoặc 'id'), 'transcript', 'duration'
            audio_id = item.get('file') or item.get('id')
            transcript = str(item['text']).lower()
            duration = item['duration']

            # Trường hợp batch đang trống (bắt đầu batch mới)
            if not current_batch["audio_paths"]:
                current_batch["audio_paths"].append(audio_id)
                current_batch["transcripts"].append(transcript)
                current_batch["total_duration"] = duration
                continue

            # Tính thử độ dài nếu thêm phần tử này (kèm khoảng lặng)
            new_duration = current_batch["total_duration"] + silence_duration + duration

            # Logic: Nếu thêm vào mà gần target hơn là không thêm, thì cộng dồn vào
            if abs(new_duration - target_duration) < abs(current_batch["total_duration"] - target_duration):
                current_batch["audio_paths"].append(audio_id)
                current_batch["transcripts"].append(transcript)
                current_batch["total_duration"] = new_duration
            else:
                # Đóng gói batch hiện tại và ghi vào file
                output_item = {
                    "combined_id": f"combined_{batch_count}",
                    "original_files": current_batch["audio_paths"],
                    "transcript": " ".join(current_batch["transcripts"]),
                    "total_duration": round(current_batch["total_duration"], 3)
                }
                f_out.write(json.dumps(output_item, ensure_ascii=False) + '\n')
                
                # Reset sang batch mới với item hiện tại
                batch_count += 1
                current_batch = {
                    "audio_paths": [audio_id],
                    "transcripts": [transcript],
                    "total_duration": duration
                }

        # Ghi batch cuối cùng nếu còn sót
        if current_batch["audio_paths"]:
            output_item = {
                "combined_id": f"combined_{batch_count}",
                "original_files": current_batch["audio_paths"],
                "transcript": " ".join(current_batch["transcripts"]),
                "total_duration": round(current_batch["total_duration"], 3)
            }
            f_out.write(json.dumps(output_item, ensure_ascii=False) + '\n')

    print(f"Xong! Đã tạo file: {output_jsonl}")
    print(f"Tổng số lượng batch sau khi gom: {batch_count + 1}")

if __name__ == "__main__":
    combine_jsonl_metadata()