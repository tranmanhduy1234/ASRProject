import json

# Tên file đầu vào và đầu ra
input_file = r'D:\chuyen_nganh\ASRProject\Data\merged_dataset_all_sorted.jsonl'
output_file = r'D:\chuyen_nganh\ASRProject\Tokenizer\text2tokenizer.txt'

try:
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            # Loại bỏ khoảng trắng thừa và kiểm tra dòng trống
            line = line.strip()
            if not line:
                continue
            
            # Giải mã JSON và lấy trường 'text'
            data = json.loads(line)
            text_content = data.get('text', '')
            
            # Ghi vào file txt, mỗi nội dung một dòng
            f_out.write(text_content + '\n')
            
    print(f"Xử lý thành công! Dữ liệu đã được ghi vào {output_file}")

except FileNotFoundError:
    print("Không tìm thấy file JSONL. Vui lòng kiểm tra lại đường dẫn.")
except Exception as e:
    print(f"Có lỗi xảy ra: {e}")