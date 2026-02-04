import json
import statistics
from collections import Counter
from pathlib import Path

def analyze_jsonl(file_path):
    """Thống kê dữ liệu từ file JSONL"""
    
    entries = []
    
    # Đọc file JSONL
    print(f"📖 Đang đọc file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line.strip())
                entries.append(entry)
            except json.JSONDecodeError as e:
                print(f"⚠️  Lỗi ở dòng {line_num}: {e}")
    
    if not entries:
        print("❌ Không có dữ liệu để phân tích!")
        return
    
    print(f"\n{'='*60}")
    print(f"📊 THỐNG KÊ TỔNG QUÁT")
    print(f"{'='*60}")
    print(f"Tổng số entries: {len(entries)}")
    
    # Thống kê từng trường
    files = [e.get('file') for e in entries]
    texts = [e.get('text') for e in entries]
    durations = [e.get('duration') for e in entries]
    sample_rates = [e.get('original_sample_rate') for e in entries]
    
    # File statistics
    print(f"\n📁 FILE:")
    print(f"  - Số entries có 'file': {sum(1 for f in files if f)}")
    print(f"  - Số entries không có 'file': {sum(1 for f in files if not f)}")
    if any(files):
        unique_files = len(set(f for f in files if f))
        print(f"  - Số file duy nhất: {unique_files}")
    
    # Text statistics
    print(f"\n📝 TEXT:")
    texts_valid = [t for t in texts if t]
    print(f"  - Số entries có 'text': {len(texts_valid)}")
    print(f"  - Số entries không có 'text': {sum(1 for t in texts if not t)}")
    
    if texts_valid:
        text_lengths = [len(str(t)) for t in texts_valid]
        word_counts = [len(str(t).split()) for t in texts_valid]
        
        print(f"  - Độ dài text (ký tự):")
        print(f"    • Min: {min(text_lengths)}")
        print(f"    • Max: {max(text_lengths)}")
        print(f"    • Avg: {statistics.mean(text_lengths):.2f}")
        print(f"    • Median: {statistics.median(text_lengths):.2f}")
        
        print(f"  - Số từ trong text:")
        print(f"    • Min: {min(word_counts)}")
        print(f"    • Max: {max(word_counts)}")
        print(f"    • Avg: {statistics.mean(word_counts):.2f}")
        print(f"    • Median: {statistics.median(word_counts):.2f}")
    
    # Duration statistics
    print(f"\n⏱️  DURATION:")
    durations_valid = [d for d in durations if d is not None and isinstance(d, (int, float))]
    print(f"  - Số entries có 'duration': {len(durations_valid)}")
    print(f"  - Số entries không có 'duration': {sum(1 for d in durations if d is None)}")
    
    if durations_valid:
        print(f"  - Giá trị duration (giây):")
        print(f"    • Min: {min(durations_valid):.2f}")
        print(f"    • Max: {max(durations_valid):.2f}")
        print(f"    • Avg: {statistics.mean(durations_valid):.2f}")
        print(f"    • Median: {statistics.median(durations_valid):.2f}")
        print(f"    • Total: {sum(durations_valid):.2f} giây ({sum(durations_valid)/60:.2f} phút)")
    
    # Sample rate statistics
    print(f"\n🎵 ORIGINAL_SAMPLE_RATE:")
    sample_rates_valid = [s for s in sample_rates if s is not None]
    print(f"  - Số entries có 'original_sample_rate': {len(sample_rates_valid)}")
    print(f"  - Số entries không có 'original_sample_rate': {sum(1 for s in sample_rates if s is None)}")
    
    if sample_rates_valid:
        rate_counter = Counter(sample_rates_valid)
        print(f"  - Sample rates khác nhau:")
        for rate, count in sorted(rate_counter.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(entries)) * 100
            print(f"    • {rate} Hz: {count} entries ({percentage:.1f}%)")
    
    print(f"\n{'='*60}")
    
    # Mẫu dữ liệu
    print(f"\n📋 MẪU DỮ LIỆU (3 entries đầu tiên):")
    print(f"{'='*60}")
    for i, entry in enumerate(entries[:3], 1):
        print(f"\nEntry {i}:")
        print(json.dumps(entry, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    # Thay đổi đường dẫn file của bạn ở đây
    file_path = r"D:\chuyen_nganh\ASRProject\merged_dataset_all_filtered.jsonl"
    
    if not Path(file_path).exists():
        print(f"❌ File không tồn tại: {file_path}")
    else:
        analyze_jsonl(file_path)