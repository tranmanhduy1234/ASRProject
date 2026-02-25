from Tokenizer.tokenizer2025 import Tokenizer2025
import torch
import torchaudio.transforms as T
from torch.nn.utils.rnn import pad_sequence
import config
import numpy as np
from Trainer.util import *
from Model.build_component.model import ASR2026
from Inference.beamsearch import BeamSearchOptim
from Data.util import *

if __name__=="__main__":
    amplitute_to_db = T.AmplitudeToDB(top_db=100)
    mel_transform = T.MelSpectrogram(
        sample_rate=config.SAMPLE_RATE,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LEN,
        n_mels=config.CHANNEL_LOG_MEL
    )
    
    audio_tensor = load_and_combine_to_tensor_sf(paths=[r"D:\chuyen_nganh\ASRProject\Inference\demo_TTS.wav"])
    mel_spec = mel_transform(audio_tensor)
    audio_mel_spectrogram = amplitute_to_db(mel_spec).to("cuda")
    
    mel_flat = audio_mel_spectrogram.transpose(1, 2)
    
    mu = mel_flat.mean()
    sigma = mel_flat.std()
    mel_flat = (mel_flat - mu) / (sigma + 1e-5)
    audio_mel_spectrogram = mel_flat.transpose(1, 2).contiguous()
    
    print(audio_mel_spectrogram.mean()) # Nên gần bằng 0
    print(audio_mel_spectrogram.std())  # Nên gần bằng 1
    
    model = ASR2026().to("cuda")
    model.eval()
    load_checkpoint_onlymodel(r"D:\chuyen_nganh\ASRProject\Save_checkpoint\checkpoint_40099_epoch_3.pt", model=model)
    print(audio_mel_spectrogram.shape)
    
    import time
    start = time.time()
    with torch.inference_mode():
        beamsearchhead = BeamSearchOptim(beam_width=5, max_len=256, sos_id=1, eos_id=2, device='cuda', alpha=0.6)
        rs, _ = beamsearchhead.batch_translate(audio_mel_spectrogram=audio_mel_spectrogram, model=model, source_mask=None, use_cache=True)
    print()
    print(f"Tổng thời gian inference: {(time.time() - start):.2f} giây")
    print()
    rs = rs.tolist()
    tokenizer2025 = Tokenizer2025(model_spm_path=r"D:\chuyen_nganh\ASRProject\Tokenizer\unigram_10000.model", legacy=False)
    print(tokenizer2025.decode(rs, skip_special_tokens=True))