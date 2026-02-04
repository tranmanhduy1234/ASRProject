import json

# Tên file đầu vào và đầu ra
input_file = r'D:\chuyen_nganh\ASRProject\Data\thivux\transcription_thivux_test.txt'
output_file = r'D:\chuyen_nganh\ASRProject\Data\thivux\transcription_thivux_test.json'

ket_qua = []

try:
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

        for index, line in enumerate(lines):
            # Xóa khoảng trắng thừa ở đầu/cuối dòng
            text_content = line.strip()
            
            # Bỏ qua dòng trống nếu cần
            if not text_content:
                print("Phát hiện dòng trống")
                continue

            # --- PHẦN QUAN TRỌNG: TẠO CÁC THUỘC TÍNH ---
            # Tại đây bạn định nghĩa các key bạn muốn
            item = {
                "id": index + 1,                  # Thuộc tính 1: Số thứ tự
                "original_text": text_content,    # Thuộc tính 2: Nội dung gốc
                "length": len(text_content),      # Thuộc tính 3: Độ dài chuỗi
                "is_processed": False             # Thuộc tính 4: Giá trị mặc định
            }
            
            # Nếu dòng text có chứa dấu phẩy để tách thông tin (VD: Tên, Tuổi)
            # Bạn có thể dùng: parts = text_content.split(',')
            # item["name"] = parts[0]
            # item["age"] = parts[1]

            ket_qua.append(item)

    # Ghi ra file JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        # ensure_ascii=False để giữ nguyên tiếng Việt có dấu
        # indent=4 để file json xuống dòng đẹp, dễ đọc
        json.dump(ket_qua, f, ensure_ascii=False, indent=4)

    print(f"Đã chuyển đổi thành công! Kiểm tra file {output_file}")

except FileNotFoundError:
    print(f"Không tìm thấy file {input_file}")