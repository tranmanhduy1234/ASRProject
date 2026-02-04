from datasets import load_dataset, Audio
import soundfile as sf
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import Counter
import unicodedata

# ============================================
# 1. CẤU HÌNH VÀ LOAD DATASET
# ============================================
NUM_SAMPLES = 20000
stats = []
vocab_counter = Counter()
errors = []

print(f"🚀 Bắt đầu phân tích {NUM_SAMPLES if NUM_SAMPLES else 'toàn bộ'} mẫu...")

ds = load_dataset("nguyendv02/ViMD_Dataset", split="train", streaming=True)
ds = ds.cast_column("audio", Audio(decode=False))

# ============================================
# 2. THU THẬP DỮ LIỆU CHI TIẾT
# ============================================
for i, item in tqdm(enumerate(ds), total=NUM_SAMPLES):
    try:
        # --- Xử lý Audio ---
        audio_bytes = item['audio']['bytes']
        with io.BytesIO(audio_bytes) as f:
            info = sf.info(f)
            duration = info.duration
            sample_rate = info.samplerate
            channels = info.channels
            
        # --- Xử lý Text ---
        text = item.get('text', "") or ""
        
        # Phân tích ký tự
        vocab_counter.update(text)
        
        # Tính toán thêm thông tin
        words = text.split()
        num_words = len(words)
        char_per_second = len(text) / duration if duration > 0 else 0
        words_per_second = num_words / duration if duration > 0 else 0
        
        # Phân loại tones trong tiếng Việt
        tonal_marks = sum(1 for c in text if unicodedata.combining(c))
        
        stats.append({
            "id": i,
            "duration": duration,
            "sample_rate": sample_rate,
            "channels": channels,
            "text_len": len(text),
            "num_words": num_words,
            "char_per_second": char_per_second,
            "words_per_second": words_per_second,
            "avg_word_len": len(text) / num_words if num_words > 0 else 0,
            "tonal_marks": tonal_marks,
            "text": text,
            "text_lower": text.lower()
        })

        if NUM_SAMPLES and i >= NUM_SAMPLES - 1:
            break

    except Exception as e:
        errors.append({"sample": i, "error": str(e)})
        continue

df = pd.DataFrame(stats)

# ============================================
# 3. PHÂN TÍCH TOÀN DIỆN
# ============================================
print("\n" + "="*60)
print("📊 BÁO CÁO PHÂN TÍCH TOÀN DIỆN DỮ LIỆU ASR TIẾNG VIỆT")
print("="*60)

# --- 1. TỔNG QUAN CHUNG ---
print(f"\n1️⃣ TỔNG QUAN CHUNG:")
print(f"   • Tổng mẫu đã phân tích: {len(df)}")
print(f"   • Tỷ lệ thành công: {len(df) / NUM_SAMPLES * 100:.2f}%")
print(f"   • Số lỗi: {len(errors)}")
total_duration_hours = df['duration'].sum() / 3600
print(f"   • Tổng thời lượng: {total_duration_hours:.2f} giờ ({df['duration'].sum() / 60:.1f} phút)")
print(f"   • Thời lượng bình quân: {df['duration'].mean():.2f} ± {df['duration'].std():.2f} giây")

# --- 2. PHÂN TÍCH AUDIO DURATION ---
print(f"\n2️⃣ PHÂN TÍCH AUDIO DURATION:")
print(f"   Min: {df['duration'].min():.3f}s | Max: {df['duration'].max():.2f}s")
print(f"   Median: {df['duration'].median():.2f}s | Q1: {df['duration'].quantile(0.25):.2f}s | Q3: {df['duration'].quantile(0.75):.2f}s")

# Phân loại khoảng thời lượng
ranges = [(0, 1), (1, 5), (5, 10), (10, 20), (20, 30), (30, 60), (60, float('inf'))]
print(f"\n   📈 Phân bố theo khoảng thời lượng:")
for start, end in ranges:
    count = len(df[(df['duration'] >= start) & (df['duration'] < end)])
    pct = count / len(df) * 100
    label = f"{start}-{end}s" if end != float('inf') else f">{start}s"
    print(f"      {label}: {count:5d} ({pct:5.1f}%)")

# Cảnh báo
n_short = len(df[df['duration'] < 0.5])
n_long = len(df[df['duration'] > 60])
print(f"\n   ⚠️ CẢNH BÁO: {n_short} mẫu < 0.5s (quá ngắn), {n_long} mẫu > 60s (quá dài)")

# --- 3. PHÂN TÍCH SAMPLE RATE ---
print(f"\n3️⃣ PHÂN TÍCH SAMPLE RATE & AUDIO FORMAT:")
sr_dist = df['sample_rate'].value_counts()
for sr, count in sr_dist.items():
    print(f"   • {sr} Hz: {count} mẫu ({count/len(df)*100:.2f}%)")
print(f"   • Channels: {df['channels'].unique()}")

# --- 4. PHÂN TÍCH TEXT & TỐC ĐỘ PHÁT ÂM ---
print(f"\n4️⃣ PHÂN TÍCH TEXT & TỐC ĐỘ PHÁT ÂM:")
print(f"   • Tổng ký tự: {df['text_len'].sum()}")
print(f"   • Chiều dài text - Min: {df['text_len'].min()} | Max: {df['text_len'].max()} | Trung bình: {df['text_len'].mean():.1f}")
print(f"   • Tổng từ: {df['num_words'].sum()}")
print(f"   • Từ bình quân/mẫu: {df['num_words'].mean():.1f} ± {df['num_words'].std():.2f}")
print(f"   • Độ dài từ bình quân: {df['avg_word_len'].mean():.2f} ký tự")

print(f"\n   📊 Tốc độ phát âm:")
print(f"      • Ký tự/giây: {df['char_per_second'].mean():.2f} ± {df['char_per_second'].std():.2f}")
print(f"      • Từ/giây: {df['words_per_second'].mean():.2f} ± {df['words_per_second'].std():.2f}")

# --- 5. PHÂN TÍCH VOCABULARY ---
print(f"\n5️⃣ PHÂN TÍCH VOCABULARY:")
unique_chars = len(vocab_counter)
print(f"   • Tổng ký tự unique: {unique_chars}")
print(f"   • Tổng ký tự (với lặp): {sum(vocab_counter.values())}")
chars_sorted = sorted(vocab_counter.keys())
print(f"   • Danh sách ký tự: {''.join(chars_sorted[:100])}")

print(f"\n   🔝 Top 20 ký tự xuất hiện nhiều nhất:")
for idx, (char, count) in enumerate(vocab_counter.most_common(20), 1):
    char_repr = repr(char) if char == ' ' else char
    print(f"      {idx:2d}. {char_repr}: {count:6d} ({count/sum(vocab_counter.values())*100:5.2f}%)")

# --- 6. PHÂN TÍCH TONAL MARKS (Dấu tone tiếng Việt) ---
print(f"\n6️⃣ PHÂN TÍCH DẤU TONE TIẾNG VIỆT:")
total_tonal = df['tonal_marks'].sum()
print(f"   • Tổng dấu tone: {total_tonal}")
print(f"   • Bình quân dấu tone/mẫu: {df['tonal_marks'].mean():.2f}")
print(f"   • Mẫu không có dấu tone: {len(df[df['tonal_marks'] == 0])}")

# --- 7. PHÂN TÍCH CORRELATION ---
print(f"\n7️⃣ PHÂN TÍCH TƯƠNG QUAN (CORRELATION):")
corr_matrix = df[['duration', 'text_len', 'num_words', 'char_per_second']].corr()
print(f"   • Correlation Duration ↔ Text Length: {corr_matrix.loc['duration', 'text_len']:.3f}")
print(f"   • Correlation Duration ↔ Num Words: {corr_matrix.loc['duration', 'num_words']:.3f}")
print(f"   • Correlation Text Length ↔ Num Words: {corr_matrix.loc['text_len', 'num_words']:.3f}")

# --- 8. PHÂN TÍCH OUTLIERS ---
print(f"\n8️⃣ PHÂN TÍCH OUTLIERS:")
Q1_dur = df['duration'].quantile(0.25)
Q3_dur = df['duration'].quantile(0.75)
IQR_dur = Q3_dur - Q1_dur
outliers_dur = df[(df['duration'] < Q1_dur - 1.5*IQR_dur) | (df['duration'] > Q3_dur + 1.5*IQR_dur)]
print(f"   • Audio Duration Outliers: {len(outliers_dur)} ({len(outliers_dur)/len(df)*100:.2f}%)")

Q1_text = df['text_len'].quantile(0.25)
Q3_text = df['text_len'].quantile(0.75)
IQR_text = Q3_text - Q1_text
outliers_text = df[(df['text_len'] < Q1_text - 1.5*IQR_text) | (df['text_len'] > Q3_text + 1.5*IQR_text)]
print(f"   • Text Length Outliers: {len(outliers_text)} ({len(outliers_text)/len(df)*100:.2f}%)")

# --- 9. PHÂN TÍCH MISSING/EMPTY VALUES ---
print(f"\n9️⃣ PHÂN TÍCH DỮ LIỆU THIẾU/RỖNG:")
empty_text = len(df[df['text_len'] == 0])
print(f"   • Mẫu text rỗng: {empty_text} ({empty_text/len(df)*100:.2f}%)")
zero_duration = len(df[df['duration'] == 0])
print(f"   • Mẫu duration = 0: {zero_duration} ({zero_duration/len(df)*100:.2f}%)")

# --- 10. GỢI Ý TIỀN XỬ LÝ ---
print(f"\n🔟 GỢI Ý TIỀN XỬ LÝ CHO MÔ HÌNH ASR:")
print(f"   ✓ Lọc bỏ: Duration < 0.5s hoặc > 60s ({len(df[(df['duration'] < 0.5) | (df['duration'] > 60)])}) mẫu")
print(f"   ✓ Lọc bỏ: Text rỗng ({empty_text} mẫu)")
print(f"   ✓ Chuẩn hóa: Sample rate thành 16kHz hoặc 8kHz")
print(f"   ✓ Dự kiến sau lọc: {len(df[(df['duration'] >= 0.5) & (df['duration'] <= 60) & (df['text_len'] > 0)])} mẫu")

# ============================================
# 4. VISUALIZATION
# ============================================
sns.set_style("whitegrid")
fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Biểu đồ 1: Phân phối Duration
ax1 = fig.add_subplot(gs[0, 0])
sns.histplot(data=df, x='duration', kde=True, bins=50, ax=ax1, color='skyblue')
ax1.set_title('Phân phối Độ dài Audio', fontsize=12, fontweight='bold')
ax1.axvline(df['duration'].mean(), color='r', linestyle='--', label=f"Mean: {df['duration'].mean():.2f}s")
ax1.legend()

# Biểu đồ 2: Phân phối Text Length
ax2 = fig.add_subplot(gs[0, 1])
sns.histplot(data=df, x='text_len', kde=True, bins=50, ax=ax2, color='orange')
ax2.set_title('Phân phối Độ dài Text', fontsize=12, fontweight='bold')

# Biểu đồ 3: Phân phối Num Words
ax3 = fig.add_subplot(gs[0, 2])
sns.histplot(data=df, x='num_words', kde=True, bins=50, ax=ax3, color='lightgreen')
ax3.set_title('Phân phối Số Từ', fontsize=12, fontweight='bold')

# Biểu đồ 4: Scatter Duration vs Text Length
ax4 = fig.add_subplot(gs[1, 0])
sns.scatterplot(data=df, x='duration', y='text_len', alpha=0.4, ax=ax4, color='purple')
ax4.set_title('Tương quan: Duration vs Text Length', fontsize=12, fontweight='bold')
ax4.set_ylabel('Text Length')

# Biểu đồ 5: Scatter Duration vs Char per Second
ax5 = fig.add_subplot(gs[1, 1])
sns.scatterplot(data=df, x='duration', y='char_per_second', alpha=0.4, ax=ax5, color='teal')
ax5.set_title('Tương quan: Duration vs Char/Second', fontsize=12, fontweight='bold')
ax5.set_ylabel('Char per Second')

# Biểu đồ 6: Boxplot Duration (Outliers)
ax6 = fig.add_subplot(gs[1, 2])
sns.boxplot(y=df['duration'], ax=ax6, color='lightcoral')
ax6.set_title('Boxplot Duration (Phát hiện Outlier)', fontsize=12, fontweight='bold')

# Biểu đồ 7: Top 15 ký tự
ax7 = fig.add_subplot(gs[2, :2])
top_chars = vocab_counter.most_common(15)
chars_labels = [c[0] if c[0] != ' ' else '[SPACE]' for c in top_chars]
chars_values = [c[1] for c in top_chars]
ax7.barh(chars_labels, chars_values, color='steelblue')
ax7.set_title('Top 15 Ký tự Xuất hiện Nhiều nhất', fontsize=12, fontweight='bold')
ax7.invert_yaxis()

# Biểu đồ 8: Phân bố Duration theo khoảng
ax8 = fig.add_subplot(gs[2, 2])
duration_bins = [(0, 1), (1, 5), (5, 10), (10, 20), (20, 30), (30, 60), (60, 1000)]
bin_labels = ['<1s', '1-5s', '5-10s', '10-20s', '20-30s', '30-60s', '>60s']
bin_counts = [len(df[(df['duration'] >= s) & (df['duration'] < e)]) for s, e in duration_bins]
ax8.bar(bin_labels, bin_counts, color='coral')
ax8.set_title('Phân bố Duration theo Khoảng', fontsize=12, fontweight='bold')
ax8.tick_params(axis='x', rotation=45)

plt.suptitle('📊 Phân Tích Chi Tiết Dataset ASR Tiếng Việt', fontsize=16, fontweight='bold', y=0.995)
plt.show()