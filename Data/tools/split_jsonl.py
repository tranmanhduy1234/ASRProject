import random

def split_jsonl_large_file(input_path, train_path, test_path, train_ratio=0.8):
    print("Đang đếm số dòng...")
    with open(input_path, 'r', encoding='utf-8') as f:
        line_count = sum(1 for _ in f)

    indices = list(range(line_count))
    random.seed(42)
    random.shuffle(indices)

    split_index = int(line_count * train_ratio)
    train_indices = set(indices[:split_index])

    print(f"Đang tách file ({line_count} dòng)...")
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(train_path, 'w', encoding='utf-8') as f_train, \
         open(test_path, 'w', encoding='utf-8') as f_test:
        
        for i, line in enumerate(f_in):
            if i in train_indices:
                f_train.write(line)
            else:
                f_test.write(line)

    print(f"Hoàn thành! Train: {split_index} dòng, Test: {line_count - split_index} dòng.")

split_jsonl_large_file(r'D:\chuyen_nganh\ASRProject\Data\combined_metadata.jsonl', 
                       r'D:\chuyen_nganh\ASRProject\Data\combined_metadata_train.jsonl', 
                       r'D:\chuyen_nganh\ASRProject\Data\combined_metadata_test.jsonl', 
                       0.95)