# ASRProject — Hệ thống nhận dạng giọng nói tiếng Việt (Vietnamese ASR)

Hệ thống **nhận dạng giọng nói tự động (ASR)** cho tiếng Việt, chuyển đổi âm thanh thành văn bản dựa trên kiến trúc encoder-decoder Transformer và Conformer-Transducer.

---

## Cài đặt

### Môi trường (Conda)

```bash
conda env create -f environment.yml
conda activate asrproject_linux
```

> **Lưu ý:** `environment.yml` được cấu hình cho Linux; trên Windows cần điều chỉnh đường dẫn nếu cần.

### Cấu hình

Chỉnh sửa `config.py` trước khi chạy:

| Tham số | Mô tả |
|---------|-------|
| `ROOT_AUDIO` | Thư mục chứa audio |
| `ADDRESS_AUDIO` | Đường dẫn file `address.jsonl` ánh xạ ID → path audio |
| `METADATA_TRAIN` / `METADATA_TEST` | File metadata huấn luyện/kiểm tra |
| `LOAD_LAST_CHECKPOINT_PATH` | Checkpoint để resume hoặc inference |
| `MODEL_SPM_PATH` | Đường dẫn tokenizer (`Tokenizer/unigram_10000.model`) |

---

## Cấu trúc dự án

```
ASRProject/
├── config.py              # Cấu hình hyperparameters, đường dẫn, audio
├── environment.yml        # Định nghĩa môi trường Conda
│
├── Data/                  # Tiền xử lý & tải dữ liệu
│   ├── dataloader2026.py  # ASRDataloader (Hugging Face, JSONL)
│   ├── util.py, downloaddata.py
│   ├── Filter/            # Lọc, sắp xếp, concat audio
│   └── tools/             # VAD, split, convert JSONL, ...
│
├── Model/                 # Mô hình ASR2026 (Transformer)
│   ├── architecture/configmodel.py
│   └── build_component/   # model, encoder, decoder, attention, ...
│
├── Model_CFMTDC/          # Conformer-Transducer (kiến trúc thay thế)
├── Tokenizer/             # Tokenizer2025 (SentencePiece, 10k vocab)
├── Trainer/               # Huấn luyện (train.py)
├── Inference/             # Chạy suy luận (run.py, beamsearch.py)
├── Save_checkpoint/       # Checkpoint (.pt)
└── runs/                  # Log TensorBoard
```

---

## Sử dụng

### Huấn luyện

```bash
python Trainer/train.py
```

- Sử dụng `METADATA_TRAIN` và `METADATA_TEST`
- Lưu checkpoint tại `Save_checkpoint/`
- Log TensorBoard tại `runs/`
- Hỗ trợ resume qua `LOAD_LAST_CHECKPOINT_PATH`

### Inference

```bash
python Inference/run.py
```

- Load checkpoint từ `config.LOAD_LAST_CHECKPOINT_PATH`
- Sử dụng beam search để decode mel-spectrogram → văn bản

---

## Chuẩn bị dữ liệu

1. **Tải dữ liệu:** `Data/downloaddata.py` — tải dataset `capleaf/viVoice` từ Hugging Face
2. **Lọc/Sắp xếp:** Các script trong `Data/Filter/` (vd: `sortJsonlByduration.py`)
3. **Ghép audio:** `Data/Filter/concatAudio.py` — tạo segment ~15 giây
4. **Chia dữ liệu:** `Data/tools/split_jsonl.py` — train/test/dev

### Định dạng metadata

**Metadata JSONL** (vd: `combined_metadata_train.jsonl`):

```json
{"combined_id": "combined_0", "original_files": ["sample_1.wav"], "transcript": "nội dung văn bản...", "total_duration": 14.5}
```

**Address JSONL** (`address.jsonl`):

```json
{"id": "sample_1.wav", "chunk": "chunk_2"}
```

---

## Thông số mô hình (ASR2026)

| Tham số | Giá trị |
|---------|---------|
| EMBED_DIM | 768 |
| NUMHEAD | 8 |
| VOCAB | 10000 |
| NUM_ENCODE | 8 |
| NUM_DECODE | 6 |
| D_FF | 1536 |
| MAXLEN | 2048 |

---

## Công nghệ chính

- **PyTorch** — deep learning
- **Transformer** — encoder-decoder ASR
- **SentencePiece** — tokenizer (unigram, 10k vocab)
- **Hugging Face** — datasets, tokenizers
- **jiwer** — đánh giá WER
- **librosa, soundfile** — xử lý audio

---

## License

Xem file LICENSE trong dự án.
