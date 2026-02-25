# ASRProject — Hệ thống nhận dạng giọng nói tiếng Việt (Vietnamese ASR)

Hệ thống **nhận dạng giọng nói tự động** cho tiếng Việt, dựa trên kiến trúc encoder-decoder Transformer và Conformer-Transducer, chuyển đổi âm thanh thành văn bản.

---

## Mục lục

- [Cài đặt](#cài-đặt)
- [Phân tích công nghệ](#phân-tích-công-nghệ)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Sử dụng](#sử-dụng)
- [Chuẩn bị dữ liệu](#chuẩn-bị-dữ-liệu)
- [Thông số mô hình](#thông-số-mô-hình)

---

## Cài đặt

```bash
conda env create -f environment.yml
conda activate asrproject_linux
```

Chỉnh sửa `config.py` để cấu hình đường dẫn dữ liệu, checkpoint và tokenizer trước khi chạy.

---

## Phân tích công nghệ

### 1. Kiến trúc mô hình

#### 1.1 ASR2026 (Transformer Encoder–Decoder)

Mô hình chính là **sequence-to-sequence** với:

- **AudioEncoderEmbedding**: Conv1d subsampling (80 mel → 768 dim)
  - 3 tầng Conv1d: kernel=3, stride 1→2→2 → giảm ~4× chiều dài
  - GroupNorm (8 groups) + GELU
  - Sinusoidal positional encoding (như Transformer gốc)
  - Khởi tạo Kaiming cho Conv, constant cho norm

- **Encoder**: 8 tầng Transformer encoder
  - Multi-head self-attention (không causal)
  - Feed-forward với GELU
  - RMSNorm + residual
  - Dropout tăng dần theo tầng (0.1 → 0.107)

- **Decoder**: 6 tầng Transformer decoder
  - Self-attention (causal) + cross-attention với encoder
  - KV cache hỗ trợ inference nhanh
  - Weight tying: output projection dùng chung trọng số với token embedding

#### 1.2 Model_CFMTDC (Conformer-Transducer)

Biến thể kiến trúc **Conformer-Transducer**:
- Conv2d subsampling cho mel-spectrogram
- Conformer blocks: multi-head self-attention + convolution module + feed-forward
- RNN-T decoder (Transducer) để giải mã trực tiếp audio→text

---

### 2. Attention và tối ưu hóa

#### 2.1 OptimizedFlashMHA (Efficient Attention)

- Dùng **`torch.nn.attention.sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION])`** — backend tối ưu của PyTorch 2.x
- `F.scaled_dot_product_attention`: memory-efficient, hỗ trợ Flash Attention khi khả dụng
- QKV fused projection (1 linear) cho self-attention
- Hỗ trợ **KV cache** để tăng tốc autoregressive decode
- Mask: causal + key padding cho sequence có độ dài thay đổi

#### 2.2 Chuẩn hóa: RMSNorm

- Dùng **RMSNorm** thay LayerNorm (như LLaMA): đơn giản hơn, nhanh hơn
- `eps=1e-6` cho ổn định số học

---

### 3. Xử lý âm thanh

| Thư viện | Mục đích |
|----------|----------|
| **torchaudio** | MelSpectrogram (n_fft=400, hop=160, 80 mels), AmplitudeToDB |
| **soundfile** | Đọc/ghi WAV, float32, đa kênh→mono |
| **librosa** | Resample audio (orig_sr → 16 kHz) |
| **numpy** | Concat waveform, silence padding giữa các clip |

**Pipeline**: WAV → resample 16 kHz → MelSpectrogram → dB (top_db=100) → chuẩn hóa (mean=0, std=1) theo batch.

---

### 4. Tokenizer và từ điển

#### 4.1 Tokenizer2025

- **LlamaTokenizerFast** (Hugging Face) với vocab từ **SentencePiece** (unigram, 10k BPE)
- Special tokens: BOS, EOS, PAD (thêm `<pad>`)
- Format input: `BOS + text + EOS`
- Batch encode/decode qua `tokenizer.batch_encode_plus` và `decode`

#### 4.2 Vocab

- Kích thước: 10,000 subword units
- File: `unigram_10000.model` + `unigram_10000.vocab`
- Phù hợp tiếng Việt (kết hợp âm tiết, dấu)

---

### 5. Pipeline dữ liệu

#### 5.1 Hugging Face Datasets

- `load_dataset('json', data_files=...)` — load JSONL metadata
- `dataset.map(preprocess_function, batched=True)` — tokenize transcript trước
- Streaming=False, keep_in_memory=False — load on-demand

#### 5.2 ASRDataloader

- **collate_fn**: padding transcript, load audio từ `address.jsonl`, tính mel, normalize
- **element_metadata_2_tensor_input_model**: ghép nhiều file WAV (có khoảng lặng 0.5s) → mel
- PyTorch DataLoader: `pin_memory`, `num_workers`, `persistent_workers`, `prefetch_factor`
- Batch: `{mel_spectrogram_dbs, mel_mask, transcripts_target_preshift, transcripts_target_shifted, transcripts_mask}`

---

### 6. Huấn luyện

| Thành phần | Công nghệ |
|------------|-----------|
| **Optimizer** | AdamW (betas=0.9/0.98, eps=1e-6, weight_decay=0.01) |
| **Scheduler** | Cosine với warmup (warmup ~5% steps, min_lr_ratio=0.2) |
| **Loss** | CrossEntropyLoss + label_smoothing=0.1, ignore PAD |
| **Mixed precision** | `torch.amp.autocast` + `GradScaler` (AMP) |
| **Gradient** | clip_grad_norm=1.0, accumulation 8 steps |
| **Checkpoint** | Lưu model, optimizer, scheduler, scaler, step, epoch |
| **TensorBoard** | Loss, LR, gradient histogram, weight/bias mean-std |

---

### 7. Inference (Beam Search)

#### 7.1 BeamSearchOptim

- Beam width: 5
- Length penalty: `((5 + len) / 6)^alpha` (alpha=0.6)
- Per-beam top-k (k = min(vocab, beam×4) để giảm chi phí
- KV cache: chỉ decode 1 token mới mỗi step khi `use_cache=True`
- Batch beam search: xử lý nhiều audio đồng thời

#### 7.2 Flow

1. `model.inference_encoder(mel)` → encoder output
2. Khởi tạo beam với BOS
3. Loop: embedding → decoder → log_softmax → top-k → cập nhật beam
4. Dừng khi tất cả beam gặp EOS
5. Chọn beam có score (có length penalty) cao nhất

---

### 8. Đánh giá

- **jiwer** — tính WER (Word Error Rate) giữa reference và hypothesis
- So sánh transcript gốc với output beam search sau khi decode

---

### 9. Công nghệ nền tảng

| Công nghệ | Phiên bản | Vai trò |
|-----------|-----------|---------|
| **PyTorch** | 2.9.1 | Deep learning, SDPA, autocast |
| **transformers** | 4.57.6 | LlamaTokenizerFast, vocab |
| **datasets** | 4.5.0 | Load JSONL, map, DataLoader |
| **accelerate** | 1.12.0 | Hạ tầng huấn luyện phân tán (nếu dùng) |
| **sentencepiece** | 0.2.1 | Nền tảng tokenizer |
| **tokenizers** | 0.22.2 | Fast tokenizer Rust backend |
| **tensorboard** | 2.20.0 | Theo dõi training |
| **jiwer** | 4.0.0 | WER |
| **librosa** | 0.11.0 | Resample audio |
| **soundfile** | 0.13.1 | Đọc/ghi âm thanh |
| **scipy, pandas, scikit-learn** | — | Phân tích, tiền xử lý dữ liệu |

---

### 10. Tóm tắt kiến trúc kỹ thuật

```
Audio WAV (16kHz)
    ↓ soundfile + librosa
Waveform
    ↓ torchaudio MelSpectrogram + AmplitudeToDB
Mel-spectrogram [B, 80, T]
    ↓ chuẩn hóa (mean, std)
    ↓ AudioEncoderEmbedding (Conv1d 3 lớp + sinusoidal PE)
[B, T/4, 768]
    ↓ 8× EncoderBlock (OptimizedFlashMHA + FFN + RMSNorm)
Encoder output
    ↓
    ├─→ Decoder (6× Block: self-attn + cross-attn + FFN)
    │       ↑
    │   Token embedding (BOS + ...) [B, L, 768]
    │
    ↓ output_projection (weight tying)
Logits [B, L, 10000]
    ↓ BeamSearchOptim
Text (decode bằng Tokenizer2025)
```

---

## Cấu trúc dự án

```
ASRProject/
├── config.py                 # Hyperparameters, đường dẫn
├── environment.yml           # Conda environment
├── Data/
│   ├── dataloader2026.py     # ASRDataloader (Hugging Face + JSONL)
│   ├── util.py               # load_audio, mel, address DB
│   ├── downloaddata.py       # Tải capleaf/viVoice từ HF
│   ├── Filter/               # concatAudio, sort, filter
│   └── tools/                # VAD, split_jsonl, ...
├── Model/
│   ├── architecture/configmodel.py
│   └── build_component/      # model, preencode, encoderblock, decoderblock,
│                             # embedding_decode, feedForwardNetword,
│                             # optimizerMultiheadAttention
├── Model_CFMTDC/             # Conformer-Transducer
├── Tokenizer/                # tokenizer2025, unigram_10000.model
├── Trainer/                  # train.py, util
├── Inference/                # run.py, beamsearch.py
├── Save_checkpoint/
└── runs/                     # TensorBoard logs
```

---

## Sử dụng

**Huấn luyện:**
```bash
python Trainer/train.py
```

**Inference:**
```bash
python Inference/run.py
```

---

## Chuẩn bị dữ liệu

1. **Tải:** `Data/downloaddata.py` — tải `capleaf/viVoice` từ Hugging Face
2. **Lọc/sắp xếp:** scripts trong `Data/Filter/`
3. **Ghép:** `Data/Filter/concatAudio.py` — tạo segment ~15s
4. **Chia:** `Data/tools/split_jsonl.py` — train/test/dev

**Metadata JSONL:**
```json
{"combined_id": "combined_0", "original_files": ["sample_1.wav"], "transcript": "...", "total_duration": 14.5}
```

**Address JSONL:**
```json
{"id": "sample_1.wav", "chunk": "chunk_2"}
```

---

## Thông số mô hình

| Tham số | Giá trị |
|---------|---------|
| EMBED_DIM | 768 |
| NUMHEAD | 8 |
| VOCAB | 10000 |
| NUM_ENCODE | 8 |
| NUM_DECODE | 6 |
| D_FF | 1536 |
| MAXLEN | 2048 |
| SAMPLE_RATE | 16000 |
| N_FFT / HOP_LEN | 400 / 160 |
| N_MELS | 80 |

## Kết quả
WER: 13.75%  
Demo:
Inference\ASR.mp4
