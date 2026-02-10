import json
import torch
import torchaudio.transforms as T
import soundfile as sf #type: ignore
from torch.nn.utils.rnn import pad_sequence
import config
import numpy as np
import librosa #type: ignore

def get_data_audio_path(target_id: str, db: dict, system="window"):
    if system == "linux":
        return config.ROOT_AUDIO + "//" + db.get(target_id)["chunk"] + "//" + db.get(target_id)["id"]
    return config.ROOT_AUDIO + "\\" + db.get(target_id)["chunk"] + "\\" + db.get(target_id)["id"]

def load_and_combine_to_tensor_sf(paths, silence_duration=0.5, target_sr=16000):
    combined_arrays = []
    
    silence_samples = int(target_sr * silence_duration)
    silence_array = np.zeros(silence_samples, dtype=np.float32)
    
    for i, path in enumerate(paths):
        try:
            data, sr = sf.read(path, dtype='float32')
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            if sr != target_sr:
                data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
            if i > 0:
                combined_arrays.append(silence_array)
            combined_arrays.append(data)
            
        except Exception as e:
            print(f"Lỗi khi load file {path}: {e}")
    if combined_arrays:
        final_array = np.concatenate(combined_arrays)
        final_tensor = torch.from_numpy(final_array).unsqueeze(0)
        return final_tensor
    return None

def load_database(file_path_input_audio_address):
    print("Đang tải jsonl đường địa chỉ dữ liệu...")
    audio_database = {}
    try:
        with open(file_path_input_audio_address, 'r', encoding='utf-8') as f_in:
            for i, line in enumerate(f_in):
                line = line.strip()
                if not line:
                    print(f"Lỗi dữ liệu file ADDRESS dòng {i + 1}")
                try:
                    data = json.loads(line)
                    
                    if "id" in data:
                        audio_database[data["id"]] = data
                    
                except json.JSONDecodeError as e:
                    print(f"Lỗi format JSON tại dòng {i}: {line[:50]}...")
                except Exception as e:
                    print(f"Lỗi không xác định tại dòng {i}: {e}")

        print(f"-> Đã load xong {len(audio_database)} phần tử đường địa chỉ vào RAM.")
    except Exception as e:
        print(e)
    return audio_database

def element_metadata_2_tensor_input_model(data_files, audio_database, mel_transform, amplitute_to_db):
    files = data_files
    paths = []
    for file in files:
        path = get_data_audio_path(file, audio_database, system="window")
        paths.append(path)
    final_waveform = load_and_combine_to_tensor_sf(paths=paths)
    mel_spectrogram = mel_transform(final_waveform)
    mel_spectrogram_db = amplitute_to_db(mel_spectrogram)
    
    return mel_spectrogram_db

if __name__=="__main__":
    file_path_metadata = config.METADATA
    file_path_input_audio_address = config.ADDRESS_AUDIO
    audio_database = load_database(file_path_input_audio_address=file_path_input_audio_address)

    with open(file_path_metadata, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            amplitute_to_db = T.AmplitudeToDB()
            mel_transform = T.MelSpectrogram(
                sample_rate=16000,
                n_fft=config.N_FFT,
                hop_length=config.HOP_LEN,
                n_mels=config.CHANNEL_LOG_MEL
            )        
            input_model = element_metadata_2_tensor_input_model(data, audio_database, mel_transform, amplitute_to_db)
            print(input_model.shape)
            print(input_model.input_features.mean()) # Nên gần bằng 0
            print(input_model.input_features.std())  # Nên gần bằng 1
            break