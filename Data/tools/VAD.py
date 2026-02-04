import torch
import soundfile as sf
import numpy as np
import torchaudio.functional as F # Dùng cái này để resample bằng toán học, ko cần codec

# 1. Tải model VAD (Giữ nguyên)
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=False,
                              trust_repo=True)

(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

# 2. Đường dẫn file
file_path = r"E:\ASR_DATASET\Audio\sample_110000016.wav"

# ==============================================================================
# PHẦN THAY ĐỔI: Dùng soundfile thay vì read_audio của silero
# ==============================================================================

# A. Đọc file bằng soundfile
# data sẽ là numpy array, samplerate là số nguyên (vd: 44100)
data, sr = sf.read(file_path)

# B. Chuyển đổi dữ liệu sang Tensor Float32
wav = torch.from_numpy(data).float()

# C. Xử lý kênh (Stereo -> Mono)
# Nếu file có nhiều kênh (shape có dạng [Time, Channels]), ta lấy trung bình cộng
if wav.ndim > 1:
    wav = wav.mean(dim=1) 

# D. Resample về 16000Hz (BẮT BUỘC CHO SILERO)
if sr != 16000:
    # F.resample nhận input shape (..., time), nên ta unsqueeze để tạo dimension giả
    # rồi sau đó squeeze lại
    wav = F.resample(wav, sr, 16000)

# ==============================================================================

# 4. Lấy các mốc thời gian (Lưu ý: sampling_rate ở đây PHẢI để 16000 vì ta đã resample ở trên)
speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=16000, threshold=0.5)

# 5. In kết quả
print(f"Tổng số đoạn giọng nói tìm thấy: {len(speech_timestamps)}")
for i, segment in enumerate(speech_timestamps):
    start_sec = segment['start'] / 16000
    end_sec = segment['end'] / 16000
    print(f"Đoạn {i+1}: Từ {start_sec:.2f}s đến {end_sec:.2f}s")