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
    
    audio_tensor = load_and_combine_to_tensor_sf(paths=[r"E:\ASR_DATASET\Audio\chunk_2\sample_310039499.wav"])
    mel_spec = mel_transform(audio_tensor)
    audio_mel_spectrogram = amplitute_to_db(mel_spec).to("cuda")
    print(audio_mel_spectrogram.mean()) # Nên gần bằng 0
    print(audio_mel_spectrogram.std())  # Nên gần bằng 1
    
    exit(0)
    model = ASR2026().to("cuda")
    model.eval()
    load_checkpoint_onlymodel(r"D:\chuyen_nganh\ASRProject\Save_checkpoint\checkpoint_39999_epoch_8.pt", model=model)
    
    with torch.no_grad():
        beamsearchhead = BeamSearchOptim(beam_width=5, max_len=256, sos_id=1, eos_id=2, device='cuda', alpha=0.6)
        rs, _ = beamsearchhead.batch_translate(audio_mel_spectrogram=audio_mel_spectrogram, model=model, source_mask=None, use_cache=True)
        
    rs = rs.tolist()
    tokenizer2025 = Tokenizer2025(model_spm_path=r"D:\chuyen_nganh\ASRProject\Tokenizer\unigram_10000.model", legacy=False)
    print(tokenizer2025.decode(rs, skip_special_tokens=True))