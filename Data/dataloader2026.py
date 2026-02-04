from datasets import load_dataset #type: ignore
from torch.utils.data import DataLoader
from Tokenizer.tokenizer2025 import Tokenizer2025
from config import *
import torch
import os
from Data.util import *

class ASRDataloader:
    def __init__(self, path_metadata, tokenizer: Tokenizer2025, 
                 mel_transform, amplitute_to_db,
                 audio_address_database):
        
        self.path_metadata = path_metadata
        self.tokenizer = tokenizer
        self.token_padding_id = self.tokenizer.get_pad_token()[0]
        self.mel_transform = mel_transform
        self.amplitute_to_db = amplitute_to_db
        self.audio_address_database = audio_address_database
        
        print(f"Load dataset from {self.path_metadata}")
        self.dataset = load_dataset(
            'json',
            data_files={'train': self.path_metadata},
            streaming=False,
            keep_in_memory=False
        )['train']
        
        print("Start pre-tokenizing dataset...")
        num_proc = os.cpu_count() if os.cpu_count() else 1
        self.dataset = self.dataset.map(
            self.preprocess_function,
            batched=True,
            num_proc=num_proc,
            desc="Tokenizing"
        )
        print("Pre-tokenization complete.")
        
    def preprocess_function(self, examples):
        example = {
            "combined_id": ["combined_0", "combined_1", "combined_2", ...], # List 1000 IDs
            "original_files": [
                ["sample_1.wav", "sample_2.wav", ...], # List của List tên file
                ["sample_11.wav", "sample_12.wav", ...],
                ...
            ],
            "transcript": [
                "làng trúc yên mình nghèo...", 
                "chẳng việc gì. em biết vậy...",
                ...
            ],
            "total_duration": [14.5, 14.5, 14.5, ...] # List 1000 số thực
        }
            
        transcript = examples["transcript"]
        encoded_ids_transcript_id, encoded_ids_transcript_piece = self.tokenizer.encode(texts=transcript)
        return {
            'transcript_id': encoded_ids_transcript_id
        }
        
    def getDataloader(self, batch_size = -1):
        if batch_size == -1:
            batch_size = BATCH_SIZE
        streamed_dataset = self.dataset
        g = torch.Generator()
        g.manual_seed(28)
        
        return DataLoader(
            streamed_dataset, 
            batch_size=batch_size, 
            collate_fn=self.collate_fn,
            pin_memory=PIN_MEMORY,
            num_workers=NUM_WORKERS,
            shuffle=SHUFFLE,
            drop_last=DROP_LAST,
            persistent_workers=PERSISTENT_WORKERS if NUM_WORKERS > 0 else False,
            prefetch_factor=PREFETCH_FATOR,
            generator=g
        )
    def collate_fn(self, batch):
        examples = [
            {
                'combined_id': 'combined_0', 
                'original_files': [...], 
                'transcript': 'làng trúc yên...', 
                'total_duration': 14.5,
                # Nếu ở preprocess_function bạn có return thêm 'input_ids' chẳng hạn, 
                # thì nó sẽ xuất hiện ở đây.
            },
            {
                'combined_id': 'combined_1', 
                'original_files': [...], 
                'transcript': 'chẳng việc gì...', 
                'total_duration': 14.5
            },
            # ... cho đến hết BATCH_SIZE
        ]
        batch_size = len(batch)
        transcripts = [element['transcript'] for element in batch]
        transcripts_ids = [element['transcript_id'] for element in batch]
        maxlen_transcripts = max(map(len, transcripts_ids))
        
        padding_token_id = self.token_padding_id
        
        transcripts_ids_padding = torch.full((batch_size, maxlen_transcripts), 
                                         fill_value=padding_token_id, 
                                         dtype=torch.int64)
        for i, arr in enumerate(transcripts_ids):
            transcripts_ids_padding[i, :len(arr)] = torch.tensor(arr, dtype=torch.int64)
            
        transcripts_mask = (transcripts_ids_padding != padding_token_id)
        transcripts_target_preshift = transcripts_ids_padding
        transcripts_target_shifted  = torch.roll(transcripts_target_preshift, shifts=-1, dims=-1)
        transcripts_target_shifted[:, -1] = padding_token_id
        
        batch_original_files = [element['original_files'] for element in batch]
        
        mel_spectrogram_dbs = []
        mel_lengths = []
        
        for original_files in batch_original_files:
            mel_spectrogram_db =  element_metadata_2_tensor_input_model(original_files, self.audio_address_database, self.mel_transform, self.amplitute_to_db)
            mel_flat = mel_spectrogram_db.transpose(1, 2).squeeze(0)
            mel_spectrogram_dbs.append(mel_flat)
            mel_lengths.append(mel_flat.size(0))
            
        mel_spectrogram_dbs = pad_sequence(mel_spectrogram_dbs, batch_first=True, padding_value=config.PADDING_MELSPECTROGRAM)
        
        max_mel_len = mel_spectrogram_dbs.size(1)
        mel_lengths_tensor = torch.tensor(mel_lengths)
        
        mel_mask = torch.arange(max_mel_len).expand(batch_size, -1) < mel_lengths_tensor.unsqueeze(1)
        
        return {
            "transcripts": transcripts, # List[str]
            "transcripts_target_preshift": transcripts_target_preshift, # Torch([batch_size, seqlen])
            "transcripts_target_shifted": transcripts_target_shifted, # Torch([batch_size, seqlen])
            "transcripts_mask": transcripts_mask, # Torch([batch_size, seqlen])
            "mel_spectrogram_dbs": mel_spectrogram_dbs.transpose(1, 2).contiguous(), # torch([batch_size, 80, time])
            "mel_mask": mel_mask # Torch([batch_size, time])
        }

    def print_config_dataloader(self):
        print(f"Config DataLoader:")
        print(f"  - Batch size: {config.BATCH_SIZE}")
        print(f"  - Num workers: {config.NUM_WORKERS}")
        print(f"  - Pin memory: {config.PIN_MEMORY}")
        print(f"  - Drop last: {config.DROP_LAST}")
        print(f"  - Persistent workers: {config.PERSISTENT_WORKERS}")
        print()

if __name__=="__main__":
    amplitute_to_db = T.AmplitudeToDB(top_db=100)
    mel_transform = T.MelSpectrogram(
        sample_rate=config.SAMPLE_RATE,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LEN,
        n_mels=config.CHANNEL_LOG_MEL
    )
    data = ASRDataloader(
        path_metadata=r"D:\chuyen_nganh\ASRProject\Data\combined_metadata_debug.jsonl",
        tokenizer=Tokenizer2025(config.MODEL_SPM_PATH),
        mel_transform=mel_transform,
        amplitute_to_db=amplitute_to_db,
        file_path_input_audio_address=config.ADDRESS_AUDIO
    )
    data.print_config_dataloader()
    datatrain = data.getDataloader(batch_size=4)
    for i, batch in enumerate(datatrain):
        print(batch["mel_mask"].shape)
        print(batch["transcripts_target_preshift"].shape)
        print(batch["transcripts_target_shifted"].shape)
        print(batch["transcripts_mask"].shape)
        print(batch["mel_spectrogram_dbs"].shape)

        exit(0)
        transcripts=batch["transcripts"]
        transcripts_target_preshift=batch["transcripts_target_preshift"]
        transcripts_target_shifted=batch["transcripts_target_shifted"]
        transcripts_mask=batch["transcripts_mask"]
        mel_spectrogram_dbs=batch["mel_spectrogram_dbs"]
        mel_mask=batch["mel_mask"]
        mel_mask_downsampled = mel_mask[:, ::4]
        
        assert mel_spectrogram_dbs.shape[:2] == mel_mask.shape, \
            f"Lỗi: Mel {mel_spectrogram_dbs.shape} không khớp với Mask {mel_mask.shape}"
            
        T_expected = (mel_spectrogram_dbs.size(1) + 3) // 4

        max_val = mel_spectrogram_dbs.max().item()
        min_val = mel_spectrogram_dbs.min().item()
        mean_val = mel_spectrogram_dbs.mean().item()
        
        if torch.isnan(mel_spectrogram_dbs).any() or torch.isinf(mel_spectrogram_dbs).any():
            print(f"!!! CẢNH BÁO: Batch {i} chứa giá trị NaN hoặc Inf")

        invalid_tokens = (transcripts_target_shifted < -100).any()
        if invalid_tokens:
            print(f"!!! CẢNH BÁO: Batch {i} chứa ID transcript không hợp lệ")

        if i == 0:
            print(f"--- Kiểm tra Batch {i} ---")
            print(f"Mel - Max: {max_val:.2f}, Min: {min_val:.2f}, Mean: {mean_val:.2f}")
            print(f"Mel Shape: {mel_spectrogram_dbs.shape}")
            print(f"Transcript Length Max: {transcripts_target_preshift.size(1)}")
            
            last_true_idx = torch.where(mel_mask[0])[0][-1].item()
            print(f"Mẫu 0: Audio kết thúc tại index {last_true_idx}, giá trị tại đó: {mel_spectrogram_dbs[0, last_true_idx, 0].item():.2f}")
            
            if last_true_idx < mel_spectrogram_dbs.size(1) - 1:
                pad_sample = mel_spectrogram_dbs[0, -1, 0].item()
                print(f"Mẫu 0: Giá trị tại vùng Padding: {pad_sample:.2f}")
    print("Hoàn tất")