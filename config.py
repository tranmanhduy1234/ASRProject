import torch

# Tham số token đặc biệt
UNKNOWN=0
BOS=1
EOS=2
PAD=3

# Tham số dataloader
PIN_MEMORY = True
NUM_WORKERS = 4
BATCH_SIZE = 16
SHUFFLE = True
DROP_LAST = False
PERSISTENT_WORKERS = True
PREFETCH_FATOR = 4

# Tham số training
USE_SCALER=True
USE_AMP=True
EPOCHS = 3
MODEL_SPM_PATH = r"D:\chuyen_nganh\ASRProject\Tokenizer\unigram_10000.model"
LOAD_LAST_CHECKPOINT_PATH = r"D:\chuyen_nganh\ASRProject\Save_checkpoint\checkpoint_39999_epoch_8.pt"
SAVE_STEP = 20000
LOGGING_STEP = 2000
DEVICES = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 1e-3
RATIO_DECAY_LEARNING_RATE = 0.2
RATIO_WARMUP_GLOBAL_STEP = 0.05
SMOOTHING = 0.1
MAX_GRAD_NORM = 1.0
SEED = 28
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.98)
EPS = 1e-6
SAVE_LAST_CHECKPOINT_PATH = r"D:\chuyen_nganh\ASRProject\Save_checkpoint\checkpoint_last.pt"
ROOT_FOLDER_SAVE_CHECKPOINT= r"D:\chuyen_nganh\ASRProject\Save_checkpoint"
ACCUMULATION_STEPS = 8 # batch_size * accumulation = batch size efficient

# Tham số inference
BEAM_WIDTH = 5
MAX_LEN_INFERENCE = 1024

# Tham số đường dẫn dữ liệu
ROOT_AUDIO = r"E:\ASR_DATASET\Audio"
ADDRESS_AUDIO = r"E:\ASR_DATASET\Audio\address.jsonl"
METADATA_TRAIN = r"D:\chuyen_nganh\ASRProject\Data\combined_metadata_train.jsonl"
METADATA_TEST = r"D:\chuyen_nganh\ASRProject\Data\combined_metadata_test.jsonl"
METADATA_DEBUG = r"D:\chuyen_nganh\ASRProject\Data\combined_metadata_debug.jsonl"
# Tham số cấu hình xử lý âm thanh
SAMPLE_RATE=16000
CHANNEL_LOG_MEL=80
N_FFT=400 # (~ 250 milisecond)
HOP_LEN=160 # (~ 10 milisecond)
PADDING_MELSPECTROGRAM=-1.0
# ========================================================================================