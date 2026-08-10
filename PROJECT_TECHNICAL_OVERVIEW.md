# ASRProject — Phân tích kiến trúc và chiều sâu kỹ thuật

## 1. Tổng quan

ASRProject là một hệ thống **nhận dạng tiếng nói tự động (Automatic Speech Recognition — ASR) cho tiếng Việt**, xây dựng gần như toàn bộ pipeline bằng PyTorch thay vì chỉ fine-tune một mô hình ASR đóng gói sẵn. Hệ thống nhận waveform, biến đổi thành log-Mel spectrogram, mã hóa đặc trưng âm học bằng Transformer encoder, rồi sinh tuần tự các subword tiếng Việt bằng Transformer decoder và beam search.

Điểm có giá trị kỹ thuật nhất của dự án không chỉ nằm ở việc “chạy được mô hình”, mà ở việc tác giả đã tự triển khai và kết nối nhiều lớp bài toán khó:

- hợp nhất hàng triệu đoạn ghi âm từ nhiều nguồn khác nhau;
- xử lý sample rate, kênh âm thanh, khoảng lặng và độ dài biến thiên;
- huấn luyện SentencePiece riêng cho tiếng Việt;
- tự xây dựng multi-head attention trên Scaled Dot-Product Attention của PyTorch;
- quản lý padding mask, causal mask và thay đổi độ dài sau subsampling;
- tối ưu huấn luyện bằng AMP, gradient accumulation, clipping, scheduler và checkpoint;
- tự xây dựng batched beam search có KV cache;
- đồng thời nghiên cứu một nhánh Conformer–Transducer độc lập.

Theo metadata đang có trong repository, tập dữ liệu sau ghép gồm **1.350.708 mẫu**, khoảng **5.292,30 giờ** âm thanh. Mô hình Transformer chính có **145.211.920 tham số có thể huấn luyện**. Đây là quy mô đủ lớn để các vấn đề về I/O, bộ nhớ GPU, ổn định số học và khả năng tái lập trở thành những bài toán kỹ thuật thực sự, không còn là chi tiết phụ.

> Phạm vi tài liệu: mô tả trạng thái code tại thời điểm khảo sát. `Model/` là pipeline đang được `Trainer/train.py` và `Inference/run.py` sử dụng. `Model_CFMTDC/` là nhánh nghiên cứu Conformer–Transducer, chưa được nối vào quy trình huấn luyện chính.

## 2. Bản đồ hệ thống

```text
Các bộ dữ liệu tiếng Việt
        │
        ▼
manifest JSONL riêng lẻ
        │  merge → lọc 1–20 giây → sắp xếp theo duration
        ▼
merged_dataset_all_sorted.jsonl (4.070.788 clip)
        │  ghép clip gần 15 giây, chèn 0,5 giây im lặng
        ▼
combined_metadata*.jsonl (1.350.708 mẫu, ~5.292 giờ)
        │
        ├──────────────► văn bản ─► SentencePiece Unigram 10.000 token
        │
        ▼
đọc WAV → mono → resample 16 kHz → nối waveform
        │
        ▼
MelSpectrogram (80 bins) → dB → chuẩn hóa từng mẫu → pad batch
        │
        ▼
Conv1d subsampling 4× + positional encoding
        │
        ▼
8 Transformer encoder blocks
        │                         BOS + subword tokens
        │                                  │
        └──────── cross-attention ◄── 8 Transformer decoder blocks
                                           │
                                           ▼
                               tied projection, vocab 10.000
                                           │
                              train: cross-entropy / infer: beam search
                                           │
                                           ▼
                                      văn bản tiếng Việt
```

## 3. Cấu trúc repository và vai trò

| Thành phần | Vai trò thực tế |
|---|---|
| `config.py` | Hyperparameter, token ID, cấu hình audio, đường dẫn dữ liệu/checkpoint |
| `Data/` | Tải, hợp nhất, lọc, ghép và nạp dữ liệu; tạo log-Mel spectrogram |
| `Tokenizer/` | Chuẩn hóa corpus, huấn luyện và sử dụng SentencePiece Unigram |
| `Model/architecture/` | Cấu hình kích thước Transformer chính |
| `Model/build_component/` | Audio frontend, attention, encoder, decoder, FFN và mô hình `ASR2026` |
| `Trainer/` | Training loop, validation, WER, TensorBoard và checkpoint |
| `Inference/` | Chạy demo và batched beam search có KV cache |
| `Model_CFMTDC/` | Các thử nghiệm Conformer, relative attention, predictor LSTM và Transducer |
| `Save_checkpoint/` | Checkpoint chứa cả model và trạng thái tối ưu hóa |

Ngoài pipeline chính, `Data/tools/` chứa các công cụ phục vụ làm sạch/biến đổi dữ liệu như VAD, dịch và chuyển đổi JSON; đây là phần hỗ trợ nghiên cứu hơn là một pipeline thống nhất có thể chạy đầu-cuối.

## 4. Dữ liệu: phần có quy mô và chi phí kỹ thuật lớn nhất

### 4.1 Quy mô quan sát trực tiếp từ repository

| Tập | Số mẫu | Tổng thời lượng | Thời lượng trung bình |
|---|---:|---:|---:|
| Toàn bộ sau ghép | 1.350.708 | 5.292,30 giờ | 14,105 giây |
| Train | 1.283.172 | 5.027,44 giờ | 14,105 giây |
| Test | 67.536 | 264,87 giờ | 14,119 giây |
| Debug | 39 | — | — |

Tỉ lệ chia xấp xỉ **95% train / 5% test**. Trước khi ghép, `merged_dataset_all_sorted.jsonl` có **4.070.788 clip**. Các file metadata trong repository chiếm hơn 1 GB, còn toàn repository khoảng 4,4 GB do có thêm checkpoint 1,7 GB.

Các thống kê trên được cộng trực tiếp từ trường `total_duration`. Chúng phản ánh metadata, không phải một cuộc kiểm kê lại toàn bộ WAV trên ổ dữ liệu ngoài; do đó chưa xác nhận được file thất lạc, audio hỏng hoặc thời lượng thực tế sau resample.

### 4.2 Chuỗi chuẩn bị dữ liệu

Pipeline chuẩn bị dữ liệu được thể hiện qua các script:

1. `Data/downloaddata.py` stream dataset từ Hugging Face, lưu từng audio thành WAV và tạo manifest chứa tên file, transcript, duration và sample rate gốc.
2. `Data/Filter/pathjsonl.py` hợp nhất manifest từ nhiều corpus tiếng Việt như infore, LSVSC, VieNeu-TTS, viMD, viVoice và VLSP.
3. `Data/Filter/filterLengOutliner.py` chỉ giữ clip có thời lượng từ 1 đến 20 giây.
4. `Data/Filter/sortJsonlByduration.py` sắp xếp clip theo thời lượng.
5. `Data/Filter/concatAudio.py` gom các clip sao cho tổng thời lượng gần 15 giây, chèn 0,5 giây im lặng giữa hai clip và nối transcript tương ứng.
6. Tập sau ghép được chia thành train/test/debug và nạp bởi `ASRDataloader`.

Ghép các câu ngắn thành đoạn gần 15 giây là một quyết định đáng chú ý. Nó giảm số lượng sample, giúp batch có độ dài tương đối đồng đều và tận dụng GPU tốt hơn. Tuy nhiên, nó cũng tạo phân phối dữ liệu nhân tạo: ranh giới giữa hai phát ngôn được biểu diễn bằng khoảng lặng cố định 0,5 giây, còn transcript chỉ được nối bằng khoảng trắng. Mô hình có thể học quy luật khoảng lặng này thay vì đầy đủ biến thiên hội thoại tự nhiên.

### 4.3 Audio frontend

Với mỗi sample, `Data/util.py` thực hiện:

- tra cứu đường dẫn vật lý từ `address.jsonl`;
- đọc WAV bằng `soundfile` dưới dạng `float32`;
- trộn nhiều kênh thành mono bằng trung bình;
- resample về 16 kHz bằng `librosa` nếu cần;
- chèn 8.000 mẫu zero, tương đương 0,5 giây ở 16 kHz, giữa các clip;
- nối waveform và tính MelSpectrogram bằng `torchaudio`;
- chuyển biên độ sang dB với `top_db=100`;
- chuẩn hóa toàn bộ ma trận Mel của từng mẫu theo z-score;
- pad các sample trong batch và tạo mask theo chiều thời gian.

Cấu hình phổ hiện tại:

| Tham số | Giá trị | Ý nghĩa |
|---|---:|---|
| Sample rate | 16.000 Hz | Chuẩn hóa nguồn audio |
| `n_fft` | 400 mẫu | Cửa sổ 25 ms ở 16 kHz |
| `hop_length` | 160 mẫu | Bước 10 ms, khoảng 100 frame/giây |
| Mel channels | 80 | Kích thước đặc trưng tại mỗi frame |
| `top_db` | 100 dB | Chặn dynamic range khi đổi sang dB |

Lưu ý: comment trong `config.py` ghi `N_FFT=400` là khoảng 250 ms, nhưng giá trị đúng ở 16 kHz là **25 ms**.

### 4.4 Tại sao tầng dữ liệu khó

Ở quy mô hơn 5.000 giờ, nút thắt không chỉ là GPU. Mỗi batch hiện đọc nhiều file nhỏ, nối NumPy array, có thể resample trên CPU rồi mới tính spectrogram. `DataLoader` dùng 4 worker, pin memory, persistent worker và prefetch 4 batch để che bớt độ trễ I/O. Việc xử lý độ dài biến thiên còn đòi hỏi hai loại padding độc lập: padding frame âm thanh và padding token transcript.

Một điểm sâu về tính đúng đắn là mask gốc có kích thước theo số frame Mel nhưng encoder convolution giảm thời gian khoảng 4 lần. `ASR2026` nội suy mask bằng nearest-neighbor về đúng độ dài encoder trước khi attention. Nếu không làm bước này, padding có thể lọt vào self-attention/cross-attention hoặc gây sai shape.

## 5. Tokenizer tiếng Việt

Tokenizer đang dùng là SentencePiece với:

- thuật toán **Unigram Language Model**;
- vocabulary 10.000 subword;
- `<unk>=0`, `<s>/BOS=1`, `</s>/EOS=2`, `<pad>=3`;
- corpus lấy từ toàn bộ transcript nguồn;
- Unicode được chuẩn hóa NFC, control character và punctuation được loại bỏ, khoảng trắng được co gọn, văn bản chuyển lowercase.

`Tokenizer2025.encode()` tự thêm BOS và EOS cho từng transcript. Dataloader dùng chuỗi hoàn chỉnh làm đầu vào decoder, sau đó dịch trái một vị trí để tạo nhãn:

```text
decoder input : [BOS, t1, t2, ..., tn, EOS]
training label: [t1,  t2, ..., tn, EOS, PAD]
```

Đây là teacher forcing tiêu chuẩn cho mô hình autoregressive. Loss bỏ qua PAD và áp dụng label smoothing 0,1.

Độ khó riêng của tiếng Việt nằm ở dấu thanh, Unicode tổ hợp, khoảng trắng giữa âm tiết và tên riêng/từ vay mượn. NFC giúp tránh một ký tự hiển thị giống nhau nhưng có nhiều biểu diễn byte. Unigram subword cân bằng giữa vocabulary theo âm tiết và khả năng biểu diễn từ hiếm. Tuy vậy, pipeline làm sạch tokenizer và transcript huấn luyện cần tuyệt đối đồng nhất; hiện transcript đưa vào dataloader không chạy lại `clean_text()`, nên punctuation có thể xuất hiện trong training dù corpus tokenizer đã loại punctuation.

## 6. Mô hình chính: Transformer encoder–decoder tự xây dựng

### 6.1 Cấu hình

| Thành phần | Cấu hình |
|---|---|
| Kích thước embedding | 768 |
| Attention heads | 8, mỗi head 96 chiều |
| Encoder blocks | 8 |
| Decoder blocks | 8 |
| FFN hidden dimension | 3.072 |
| Vocabulary | 10.000 |
| Decoder max length | 2.048 token |
| Dropout | tăng nhẹ từ 0,100 đến 0,107 theo độ sâu |
| Tổng tham số trainable | 145.211.920 |
| Audio frontend | 3.730.176 tham số |
| 8 encoder blocks | 56.663.040 tham số |
| 8 decoder blocks | 75.555.840 tham số |
| Token embedding + output bias | 9.262.864 tham số duy nhất |

Các nhóm trên được đếm theo parameter duy nhất. `output_projection.weight` được trỏ trực tiếp tới `token_embed.weight`, nên ma trận embedding 10.000 × 768 chỉ xuất hiện một lần trong tổng số dù được dùng ở cả đầu vào lẫn đầu ra.

### 6.2 Audio embedding và subsampling

Đầu vào `[B, 80, T]` đi qua ba Conv1d:

```text
80 → 768, kernel 3, stride 1
768 → 768, kernel 3, stride 2
768 → 768, kernel 3, stride 2
```

Mỗi lớp dùng GroupNorm 8 nhóm và GELU. Hai stride 2 giảm chuỗi thời gian khoảng 4 lần; đoạn 15 giây từ khoảng 1.500 frame còn khoảng 375 vị trí. Đây là tối ưu rất quan trọng vì self-attention có chi phí thời gian/bộ nhớ bậc hai theo độ dài: giảm `T` bốn lần có thể giảm ma trận attention xấp xỉ 16 lần.

Sau convolution, mã hóa vị trí sinusoidal được cộng vào đặc trưng. Conv được khởi tạo Kaiming, GroupNorm bắt đầu với scale 1 và bias 0.

### 6.3 Encoder block

Mỗi encoder block là kiến trúc pre-norm:

```text
x ─► RMSNorm ─► self-attention ─► dropout ─► +x
  └──────────────────────────────────────────┘
       ─► RMSNorm ─► FFN(GELU) ─► dropout ─► residual
```

RMSNorm giúp chuẩn hóa biên độ activation với phép tính đơn giản hơn LayerNorm. FFN mở rộng 768 lên 3.072 rồi chiếu về 768. Residual connection và pre-norm là các lựa chọn quan trọng để gradient đi qua mô hình sâu ổn định hơn.

### 6.4 Attention tối ưu

`OptimizedFlashMHA` không bọc `nn.MultiheadAttention`; nó tự quản lý:

- ma trận QKV fused kích thước `3D × D` cho self-attention;
- tách projection Q, K, V khi cross-attention;
- reshape thành `[batch, head, time, head_dim]`;
- kết hợp causal mask và key-padding mask;
- gọi `scaled_dot_product_attention` với backend `EFFICIENT_ATTENTION`;
- lưu/reuse KV cache trong inference.

Về mặt toán học, mỗi head tính:

```text
Attention(Q, K, V) = softmax(QKᵀ / √d_head + mask)V
```

Self-attention encoder nhìn hai chiều trên toàn bộ tín hiệu. Self-attention decoder dùng causal mask để token ở vị trí `u` không thấy tương lai. Cross-attention cho decoder truy cập toàn bộ biểu diễn âm học hợp lệ.

Việc tự triển khai tầng này có độ khó cao vì chỉ một sai lệch nhỏ trong semantics của boolean mask, chiều tensor, cache hoặc causal condition cũng có thể làm mô hình vẫn chạy nhưng học sai. Đây là dạng lỗi khó phát hiện hơn lỗi shape thông thường.

### 6.5 Decoder và weight tying

Mỗi decoder block gồm ba nhánh pre-norm:

1. causal self-attention trên transcript đã sinh;
2. cross-attention từ token sang encoder output;
3. feed-forward network.

Token embedding được nhân `√768` rồi cộng learned positional embedding. Sau 8 block, linear projection tạo logits trên 10.000 token. Trọng số projection và token embedding được chia sẻ (**weight tying**), giúp giảm khoảng 7,68 triệu tham số độc lập và ép không gian biểu diễn đầu vào/đầu ra có quan hệ nhất quán.

### 6.6 Độ phức tạp

Gọi `T` là số frame sau subsampling, `U` là số token đích, `D=768`:

- encoder self-attention: `O(T²D)`;
- decoder self-attention: `O(U²D)` khi train;
- cross-attention: `O(TUD)`;
- FFN: xấp xỉ `O((T+U)D·D_ff)`;
- beam search không cache phải chạy lại lịch sử token, còn cache đưa phần projection attention theo thời gian gần hơn về tăng tuyến tính theo từng bước, dù vẫn phải chấm điểm vocabulary và quản lý beam.

Với audio dài, `T²` là nguyên nhân chính buộc phải subsample hoặc bucket theo duration. Với vocabulary 10.000, output logits và beam expansion cũng tiêu tốn đáng kể bộ nhớ/băng thông.

## 7. Huấn luyện và ổn định số học

### 7.1 Objective

Pipeline chính là attention-based encoder–decoder, tối ưu cross-entropy theo token:

```text
L = CrossEntropy(logits, shifted_target,
                 ignore_index=PAD,
                 label_smoothing=0.1)
```

Đây **không phải** CTC loss hay RNN-T loss. Alignment audio–text được học ngầm qua cross-attention và teacher forcing.

### 7.2 Chiến lược tối ưu hóa

- optimizer AdamW, learning rate cực đại `1e-3`;
- `betas=(0.9, 0.98)`, `eps=1e-6`, weight decay `0.01`;
- cosine decay có warmup khoảng 5% số optimizer update và sàn learning rate 20%;
- automatic mixed precision và `GradScaler`;
- gradient accumulation 8 micro-batch;
- clip global gradient norm ở 1,0;
- kiểm tra loss NaN/Inf và dừng khi phát hiện lỗi;
- seed Python, NumPy, CPU/CUDA PyTorch; bật deterministic CuDNN;
- log loss, learning rate, histogram/RMS gradient và thống kê weight vào TensorBoard.

Với batch mặc định 16 và accumulation 8, batch hiệu dụng theo cấu hình là 128 sample; tuy nhiên `Trainer2026` hiện gọi dataloader với `batch_size=4`, nên batch hiệu dụng thực tế của đường chạy đó là **32 sample**.

AMP là cần thiết cho mô hình 145 triệu tham số, nhưng tạo thêm yêu cầu về loss scaling. Code chỉ gọi scheduler khi optimizer step không bị scaler bỏ qua, thể hiện sự xử lý đúng một edge case quan trọng: không tiến learning-rate schedule khi gradient overflow khiến update không xảy ra.

### 7.3 Checkpoint và quan sát mô hình

Checkpoint lưu:

- model state;
- optimizer state;
- scheduler state;
- GradScaler state;
- step và epoch.

Checkpoint `checkpoint_40099_epoch_3.pt` khoảng 1,7 GB. Kích thước lớn hơn nhiều so với riêng trọng số model là hợp lý vì AdamW giữ thêm first/second moments và checkpoint còn mang trạng thái huấn luyện. Cách lưu này hỗ trợ resume đúng động lực optimizer, khác với chỉ nạp weight để inference.

## 8. Suy luận: batched beam search và KV cache

`BeamSearchOptim` mã hóa audio một lần, nhân encoder output cho `B=5` beam, rồi lặp autoregressive tối đa 1.024 token theo cấu hình chung (demo giới hạn 256).

Ở mỗi bước:

1. decoder trả logits token kế tiếp;
2. chuyển thành log-probability;
3. với mỗi beam chỉ giữ `min(vocab_size, beam_width × 4)` token tốt nhất;
4. gộp ứng viên của mọi parent beam và chọn lại 5 beam tốt nhất;
5. reorder KV cache theo parent beam mới;
6. beam đã EOS chỉ được phép tiếp tục EOS;
7. khi kết thúc, áp dụng length penalty `((5 + length) / 6)^0.6`.

Nếu bật cache, decoder self-attention chỉ nhận token mới nhất, nối K/V mới vào cache. Cross-attention cache K/V của encoder để không projection lại ở từng token. Reorder cache là phần đặc biệt khó: sau mỗi lần top-k, beam mới có thể đến từ parent bất kỳ; cache của mọi decoder layer phải được index lại đúng cùng thứ tự.

## 9. Nhánh Conformer–Transducer

`Model_CFMTDC/` cho thấy hướng nghiên cứu thứ hai:

- Conv2d subsampling hoặc module subsampling riêng;
- Conformer block với hai FFN “macaron”, relative positional MHSA và convolution depthwise;
- predictor LSTM trên lịch sử token;
- joiner kết hợp trạng thái âm học `f_t` và trạng thái ngôn ngữ `g_u` thành logits `[B,T,U,V]`.

Về ý tưởng, RNN-T phù hợp streaming ASR hơn encoder–decoder attention vì nó học alignment monotonic và cho phép phát token theo thời gian. Nhưng nhánh hiện tại mới ở mức prototype:

- không có trainer sử dụng RNN-T loss;
- chưa có decoder Transducer hoàn chỉnh;
- có nhiều phiên bản lớp trùng ý tưởng trong cùng thư mục;
- `r_mhsa.py` còn lệnh debug `print(attn_mask)` và phụ thuộc biến chỉ tồn tại khi mask khác `None`;
- API, special token và kích thước vocabulary chưa thống nhất với pipeline chính.

Vì vậy, Conformer–Transducer nên được trình bày như chiều sâu nghiên cứu/thiết kế mở rộng, không phải mô hình đã huấn luyện và đánh giá ngang hàng với `ASR2026`.

## 10. Những điểm khó nhất về mặt kỹ thuật

### 10.1 Quản trị dữ liệu đa nguồn ở quy mô hàng triệu file

Khác biệt sample rate, số kênh, chất lượng transcript, convention đường dẫn và file hỏng đều có thể làm pipeline dừng hoặc âm thầm giảm chất lượng. Việc ghép hơn 4 triệu clip thành hơn 1,35 triệu sequence đòi hỏi giữ đồng bộ tuyệt đối giữa thứ tự waveform và transcript.

### 10.2 Alignment không được cung cấp trực tiếp

Mô hình chỉ nhận cặp audio–transcript, không có timestamp theo token. Cross-attention phải tự học quan hệ giữa hàng trăm frame âm học và hàng chục subword. Tiếng Việt làm bài toán khó hơn do thanh điệu, đồng âm, phát âm vùng miền và ranh giới “từ” không trùng hoàn toàn với khoảng trắng.

### 10.3 Mask xuyên suốt nhiều miền độ dài

Pipeline có ít nhất ba miền độ dài: waveform, frame Mel, frame sau Conv1d và token. Mask phải được tạo theo Mel, biến đổi theo subsampling, broadcast sang attention head và kết hợp causal mask ở decoder. Sai một convention `True=valid`/`True=masked` có thể không gây exception nhưng làm chất lượng sụp đổ.

### 10.4 Tối ưu bộ nhớ và throughput

145 triệu tham số, attention bậc hai và hàng triệu file audio đòi hỏi phối hợp subsampling, AMP, accumulation, efficient SDPA, prefetch, pin memory và cache. Tối ưu từng phần riêng lẻ chưa đủ; I/O CPU, RAM, VRAM và compute GPU phải cân bằng.

### 10.5 Beam search có trạng thái

Vector hóa beam search theo batch đã khó; thêm KV cache làm trạng thái của từng beam trải trên cả 8 decoder layer. Mỗi lần beam đổi parent đều phải reorder tất cả cache. Đây là phần code thuật toán có độ sâu cao nhất trong đường inference.

### 10.6 Khả năng tái lập một thí nghiệm dài

Training hàng nghìn giờ dữ liệu có chi phí lớn nên checkpoint phải khôi phục cả optimizer/scheduler/scaler và chính xác vị trí trong epoch. Một sai lệch nhỏ ở scheduler step, shuffle order hay resume index có thể làm lần chạy tiếp không tương đương lần chạy liên tục.

## 11. Đánh giá trung thực trạng thái kỹ thuật

### 11.1 Điểm mạnh

- Pipeline ASR end-to-end được triển khai ở mức module, không phụ thuộc mô hình pretrained.
- Quy mô metadata lớn và có chiến lược ghép/batching thực dụng.
- Attention, mask, weight tying và cache được tự xây dựng, thể hiện hiểu biết sâu về Transformer.
- Có các kỹ thuật training cần thiết cho mô hình lớn: AMP, clipping, accumulation, warmup/decay, checkpoint đầy đủ.
- Có quan sát gradient/weight chi tiết bằng TensorBoard.
- Có hướng mở rộng Conformer–Transducer, cho thấy dự án không dừng ở một kiến trúc duy nhất.

### 11.2 Hạn chế và rủi ro hiện tại

1. **Portability thấp.** Nhiều đường dẫn Windows được hard-code trong `config.py` và script dữ liệu; `get_data_audio_path()` mặc định tạo đường dẫn Windows. Demo inference đã được chỉnh cục bộ sang Linux nhưng cấu hình huấn luyện chưa đồng nhất.
2. **Luồng train/resume bị chặn bởi evaluation.** Trong `Trainer2026.__init__`, nếu checkpoint model nạp và chạy WER thành công, code gọi `exit(0)` trước khi tạo optimizer và resume training.
3. **WER bị thiên lệch.** Hàm `WER_f` bỏ qua sample có WER lớn hơn 1 rồi lấy trung bình WER từng câu. Cách này đánh giá lạc quan hơn dữ liệu thật và không phải corpus-level WER chuẩn.
4. **KV cache sai positional index tiềm ẩn.** Khi cache bật, beam search chỉ đưa token cuối vào `Embedding_Decode`; lớp này luôn tạo position bắt đầu từ 0. Do đó token ở các bước sau có thể đều nhận learned positional embedding vị trí 0.
5. **Ép backend attention.** Chỉ yêu cầu `SDPBackend.EFFICIENT_ATTENTION` có thể gây lỗi hoặc giảm tính portable trên thiết bị/PyTorch không hỗ trợ backend đó; chưa có fallback rõ ràng cho CPU/math SDPA.
6. **Xử lý audio lỗi chưa đủ phòng vệ.** Hàm đọc audio in lỗi rồi bỏ clip; nếu mọi clip của sample đều lỗi, kết quả `None` vẫn được đưa vào MelSpectrogram.
7. **Text normalization không thống nhất.** Corpus tokenizer được NFC/lowercase/bỏ punctuation, nhưng transcript trong dataloader không áp dụng cùng một hàm chuẩn hóa trước encode.
8. **Checkpoint path dùng dấu `\`.** Ghép `rootfoldersave + "\checkpoint..."` không portable; trên Linux ký tự backslash có thể trở thành một phần tên file thay vì separator.
9. **Cấu hình và tài liệu cũ lệch code.** README nói 6 decoder layer trong một đoạn, nhưng `configmodel.py` đang đặt 8. Batch size cấu hình 16 nhưng trainer truyền trực tiếp 4.
10. **Thiếu test tự động.** Chưa có unit test cho mask, cache equivalence, shape sau subsampling, tokenizer round-trip, resume checkpoint hoặc beam search.
11. **Nhánh Transducer chưa hoàn thiện.** Chưa có loss, training/inference integration và benchmark để kết luận hiệu quả.
12. **Khả năng tái lập môi trường có rủi ro.** `environment.yml` đang có dấu hiệu được lưu ở UTF-16 và chứa các phiên bản rất cụ thể; công cụ Conda thông thường có thể không đọc file như YAML UTF-8 chuẩn.

Các hạn chế này không làm mất giá trị kỹ thuật của dự án; ngược lại, chúng chỉ ra đúng những bước cần hoàn thiện để chuyển từ prototype nghiên cứu quy mô lớn sang hệ thống đáng tin cậy.

## 12. Mức độ trưởng thành theo thành phần

| Thành phần | Mức hiện tại | Nhận xét |
|---|---|---|
| Thu thập/hợp nhất dữ liệu | Prototype quy mô lớn | Có dữ liệu thực, nhưng nhiều script/path thủ công |
| Audio preprocessing | Hoạt động | Đầy đủ resample/Mel/mask; cần cache và error policy |
| Tokenizer | Hoạt động | SentencePiece riêng; cần thống nhất normalization |
| Transformer training | Hoạt động | Đã có checkpoint lớn và training stack tương đối đầy đủ |
| Transformer inference | Hoạt động có điều kiện | Có beam + cache; cần test correctness và portable device |
| Evaluation | Chưa đáng tin cậy hoàn toàn | Cần sửa WER và công bố benchmark chuẩn |
| Conformer–Transducer | Research prototype | Kiến trúc có chiều sâu nhưng chưa thành pipeline end-to-end |
| Reproducibility/CI | Thấp | Hard-coded path, chưa có test và CLI/config chuẩn hóa |
| Production serving | Chưa triển khai | Chưa có streaming API, export, quantization hoặc monitoring |

## 13. Lộ trình kỹ thuật đề xuất

### Ưu tiên 1 — Chứng minh tính đúng đắn

- viết test so sánh logits decode toàn chuỗi với decode từng token có cache;
- truyền offset vị trí đúng vào decoder embedding khi dùng KV cache;
- test mask với batch có độ dài khác nhau;
- sửa WER thành corpus-level WER, không loại sample khó;
- tách rõ chế độ `train`, `resume`, `evaluate`, `infer` bằng CLI.

### Ưu tiên 2 — Tái lập và portability

- thay hard-coded path bằng `pathlib`, YAML/TOML hoặc argument CLI;
- chuẩn hóa `environment.yml` về UTF-8;
- cố định schema metadata và validate trước training;
- lưu config/model version vào checkpoint;
- thêm smoke test CPU và GPU, fallback SDPA backend.

### Ưu tiên 3 — Hiệu năng dữ liệu

- tiền tính và lưu log-Mel hoặc dùng định dạng shard như WebDataset/Arrow;
- bucket theo duration để giảm padding;
- đo profile tỉ lệ thời gian I/O, resample, Mel và GPU compute;
- cân nhắc SpecAugment, speed perturbation và noise augmentation thay vì chỉ nối silence cố định.

### Ưu tiên 4 — Chất lượng mô hình

- xây validation split không trùng speaker/domain với train;
- báo cáo WER theo nguồn dữ liệu, độ dài, vùng giọng và audio tự nhiên/TTS;
- thêm CTC auxiliary loss cho encoder hoặc hoàn thiện RNN-T loss;
- benchmark greedy, beam, beam + language model và cache on/off;
- đánh giá checkpoint bằng CER bên cạnh WER vì tiếng Việt có ranh giới từ đặc thù.

### Ưu tiên 5 — Hoàn thiện Conformer–Transducer

- hợp nhất một implementation Conformer duy nhất;
- định nghĩa blank token tách biệt và đồng bộ vocabulary;
- tích hợp RNNT loss tối ưu;
- viết greedy/beam transducer decoder có trạng thái predictor;
- so sánh latency streaming, WER và số tham số với Transformer chính.

## 14. Kết luận

ASRProject là một dự án ASR tiếng Việt có **độ khó cao và chiều sâu kỹ thuật rõ ràng**. Giá trị cốt lõi nằm ở phạm vi end-to-end và mức độ tự chủ: từ hàng triệu clip thô, tokenizer riêng, frontend âm học, Transformer 145 triệu tham số, training mixed precision đến beam search có cache. Đây là khối lượng công việc giao thoa giữa xử lý tín hiệu số, NLP, deep learning systems và data engineering.

Ở trạng thái hiện tại, dự án phù hợp nhất với định nghĩa **research engineering prototype quy mô lớn**: đã có mô hình chính, checkpoint và đầy đủ mảnh ghép quan trọng, nhưng cần củng cố correctness test, evaluation, portability và cấu hình hóa trước khi có thể coi là một hệ thống ASR production-ready. Nhánh Conformer–Transducer làm tăng chiều sâu nghiên cứu, song cần được hoàn thiện bằng loss, decoder và benchmark thực nghiệm để trở thành một đóng góp vận hành được.
