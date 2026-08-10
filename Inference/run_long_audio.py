"""Nhan dang mot file WAV dai bang mo hinh ASR2026.

Pipeline duoc thiet ke de khong nap toan bo audio vao RAM:

1. Doc WAV theo luong.
2. Chia thanh cac cua so toi da 15 giay.
3. Uu tien cat tai khoang lang gan cuoi cua so.
4. Neu buoc phai cat giua tieng noi, tao overlap va loai phan text lap.

Co the dien truc tiep ``AUDIO_PATH_OR_URL`` va cac bien duong dan o dau file,
sau do chay tu thu muc goc cua du an:

    python Inference/run_long_audio.py

Hoac truyen/ghi de bang tham so CLI:

    python Inference/run_long_audio.py --audio /duong/dan/audio_dai.wav

``--audio`` cung chap nhan URL HTTP/HTTPS tro truc tiep den file WAV.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T
from torch.nn.utils.rnn import pad_sequence

# Cho phep chay ca hai kieu:
#   python Inference/run_long_audio.py ...
#   python -m Inference.run_long_audio ...
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from Inference.beamsearch import BeamSearchOptim  # noqa: E402
from Model.build_component.model import ASR2026  # noqa: E402
from Tokenizer.tokenizer2025 import Tokenizer2025  # noqa: E402


# ============================================================================
# CAU HINH DUONG DAN TRUC TIEP TAI DAY
#
# Dien AUDIO_PATH_OR_URL roi chay:
#     python Inference/run_long_audio.py
#
# Cac tham so CLI van hoat dong va se ghi de cac gia tri ben duoi.
# AUDIO_PATH_OR_URL chap nhan ca local path va URL HTTP/HTTPS.
# ============================================================================
AUDIO_PATH_OR_URL = r"/home/tranmanhduy/Workspace/chuyen_nganh/ASRProject/Inference/demo.wav"
CHECKPOINT_PATH = PROJECT_ROOT / "Save_checkpoint" / "checkpoint_40099_epoch_3.pt"
TOKENIZER_PATH = PROJECT_ROOT / "Tokenizer" / "unigram_10000.model"

# Dat la None neu chi muon in ket qua ra terminal.
OUTPUT_TEXT_PATH: Path | None = PROJECT_ROOT / "Inference" / "long_audio_transcript.txt"
SEGMENTS_JSON_PATH: Path | None = PROJECT_ROOT / "Inference" / "long_audio_segments.json"


@dataclass
class AudioChunk:
    index: int
    start_seconds: float
    end_seconds: float
    waveform: np.ndarray
    overlaps_previous: bool
    dbfs: float


@dataclass
class SegmentResult:
    index: int
    start_seconds: float
    end_seconds: float
    text: str
    overlaps_previous: bool
    dbfs: float
    score: float | None
    skipped_as_silence: bool = False


def dbfs(waveform: np.ndarray) -> float:
    """RMS theo dBFS, voi audio float co full scale la 1.0."""
    if waveform.size == 0:
        return -math.inf
    rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
    return 20.0 * math.log10(max(rms, 1e-12))


def find_silence_cut(
    waveform: np.ndarray,
    sample_rate: int,
    search_seconds: float,
    silence_threshold_db: float,
) -> tuple[int, bool]:
    """Tim frame nang luong thap nhat o cuoi cua so.

    Tra ve ``(cut_sample, found_silence)``. Neu khong co khoang du im,
    cut_sample nam o cuoi waveform va caller se dung overlap.
    """
    if search_seconds <= 0:
        return waveform.size, False

    search_samples = min(waveform.size, int(round(search_seconds * sample_rate)))
    search_start = waveform.size - search_samples
    frame_size = max(1, int(round(0.03 * sample_rate)))
    hop_size = max(1, frame_size // 2)
    tail = waveform[search_start:]

    if tail.size < frame_size:
        return waveform.size, False

    frames = np.lib.stride_tricks.sliding_window_view(tail, frame_size)[::hop_size]
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    best_frame = int(np.argmin(rms))
    best_dbfs = 20.0 * math.log10(max(float(rms[best_frame]), 1e-12))

    window_dbfs = dbfs(waveform)
    # Voi file co volume nho, nguong dBFS tuyet doi co the coi nham mot frame
    # speech la silence. Yeu cau frame cat phai thap hon RMS cua ca cua so 8 dB.
    # Ngoai le la cua so gan nhu im hoan toan (< -65 dBFS).
    adaptive_threshold = min(silence_threshold_db, window_dbfs - 8.0)
    if window_dbfs > -65.0 and best_dbfs > adaptive_threshold:
        return waveform.size, False

    # Cat o giua frame im de de lai mot it silence o hai phia.
    cut_sample = search_start + best_frame * hop_size + frame_size // 2
    return max(1, min(cut_sample, waveform.size)), True


def iter_audio_chunks(
    audio_path: Path,
    chunk_seconds: float,
    overlap_seconds: float,
    silence_search_seconds: float,
    silence_threshold_db: float,
) -> Iterator[AudioChunk]:
    """Doc audio theo luong va yield tung cua so mono tai sample rate goc."""
    with sf.SoundFile(audio_path) as audio_file:
        sample_rate = int(audio_file.samplerate)
        max_samples = int(round(chunk_seconds * sample_rate))
        overlap_samples = int(round(overlap_seconds * sample_rate))

        buffer = np.empty(0, dtype=np.float32)
        buffer_start = 0
        chunk_index = 0
        overlaps_previous = False
        reached_eof = False

        while buffer.size > 0 or not reached_eof:
            while buffer.size < max_samples and not reached_eof:
                frames_needed = max_samples - buffer.size
                block = audio_file.read(
                    frames=frames_needed,
                    dtype="float32",
                    always_2d=True,
                )
                if block.shape[0] == 0:
                    reached_eof = True
                    break
                mono = block.mean(axis=1, dtype=np.float32)
                buffer = np.concatenate((buffer, mono))

            if buffer.size == 0:
                break

            is_full_window = buffer.size >= max_samples
            if is_full_window:
                cut_sample, found_silence = find_silence_cut(
                    buffer[:max_samples],
                    sample_rate,
                    silence_search_seconds,
                    silence_threshold_db,
                )
            else:
                cut_sample, found_silence = buffer.size, True

            waveform = np.ascontiguousarray(buffer[:cut_sample])
            start_seconds = buffer_start / sample_rate
            end_seconds = (buffer_start + cut_sample) / sample_rate
            yield AudioChunk(
                index=chunk_index,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                waveform=waveform,
                overlaps_previous=overlaps_previous,
                dbfs=dbfs(waveform),
            )
            chunk_index += 1

            if not is_full_window:
                break

            # Silence-aware cut thi khong can overlap. Hard cut dung overlap de
            # model duoc nghe lai phan tu co the bi chia doi tai bien.
            use_overlap = not found_silence and overlap_samples > 0
            actual_overlap = min(overlap_samples, max(0, cut_sample - 1)) if use_overlap else 0
            consumed_samples = cut_sample - actual_overlap
            buffer = buffer[consumed_samples:]
            buffer_start += consumed_samples
            overlaps_previous = actual_overlap > 0


class LongAudioTranscriber:
    def __init__(
        self,
        checkpoint_path: Path,
        tokenizer_path: Path,
        device: torch.device,
        beam_width: int,
        max_decode_length: int,
        use_amp: bool,
    ) -> None:
        self.device = device
        self.use_amp = use_amp and device.type == "cuda"

        self.mel_transform = T.MelSpectrogram(
            sample_rate=config.SAMPLE_RATE,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LEN,
            n_mels=config.CHANNEL_LOG_MEL,
        )
        self.amplitude_to_db = T.AmplitudeToDB(top_db=100)
        self.resamplers: dict[int, T.Resample] = {}

        print(f"Dang load model tren {device} ...")
        self.model = ASR2026().to(device)
        # Load checkpoint tren CPU de tranh giu them mot ban sao 1.7 GB tren VRAM.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.tokenizer = Tokenizer2025(model_spm_path=str(tokenizer_path), legacy=False)
        self.beam_search = BeamSearchOptim(
            beam_width=beam_width,
            max_len=max_decode_length,
            sos_id=config.BOS,
            eos_id=config.EOS,
            device=str(device),
            alpha=0.6,
        )

    def _resample(self, waveform: np.ndarray, source_rate: int) -> np.ndarray:
        if source_rate == config.SAMPLE_RATE:
            return waveform

        # Cache filter resample de khong tinh lai kernel cho moi cua so.
        if source_rate not in self.resamplers:
            self.resamplers[source_rate] = T.Resample(
                orig_freq=source_rate,
                new_freq=config.SAMPLE_RATE,
            )
        source = torch.from_numpy(waveform).unsqueeze(0)
        return self.resamplers[source_rate](source).squeeze(0).numpy()

    def _features(self, chunks: Sequence[AudioChunk], source_rate: int) -> tuple[torch.Tensor, torch.Tensor]:
        feature_list: list[torch.Tensor] = []
        feature_lengths: list[int] = []

        for chunk in chunks:
            waveform = self._resample(chunk.waveform, source_rate)
            waveform_tensor = torch.from_numpy(waveform).unsqueeze(0)
            mel = self.amplitude_to_db(self.mel_transform(waveform_tensor))
            mel_time_first = mel.transpose(1, 2).squeeze(0)

            # Giong cach normalize trong dataloader luc train: normalize rieng
            # tung sample va chi pad sau khi normalize.
            mean = mel_time_first.mean()
            std = mel_time_first.std()
            mel_time_first = (mel_time_first - mean) / (std + 1e-5)
            feature_list.append(mel_time_first)
            feature_lengths.append(mel_time_first.size(0))

        padded = pad_sequence(
            feature_list,
            batch_first=True,
            padding_value=config.PADDING_MELSPECTROGRAM,
        )
        lengths = torch.tensor(feature_lengths, dtype=torch.long)
        mask = torch.arange(padded.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
        return padded.transpose(1, 2).contiguous(), mask

    @torch.inference_mode()
    def transcribe_batch(self, chunks: Sequence[AudioChunk], source_rate: int) -> list[SegmentResult]:
        features, source_mask = self._features(chunks, source_rate)
        features = features.to(self.device, non_blocking=self.device.type == "cuda")
        source_mask = source_mask.to(self.device, non_blocking=self.device.type == "cuda")

        amp_context = torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.use_amp,
        )
        with amp_context:
            token_ids, scores = self.beam_search.batch_translate(
                audio_mel_spectrogram=features,
                model=self.model,
                source_mask=source_mask,
                use_cache=True,
            )

        texts = self.tokenizer.decode(token_ids.detach().cpu().tolist(), skip_special_tokens=True)
        score_values = scores.detach().float().cpu().tolist()
        return [
            SegmentResult(
                index=chunk.index,
                start_seconds=chunk.start_seconds,
                end_seconds=chunk.end_seconds,
                text=text.strip(),
                overlaps_previous=chunk.overlaps_previous,
                dbfs=chunk.dbfs,
                score=float(score),
            )
            for chunk, text, score in zip(chunks, texts, score_values)
        ]


def normalize_word(word: str) -> str:
    return re.sub(r"[^\w]", "", word, flags=re.UNICODE).casefold()


def number_of_repeated_prefix(previous_words: list[str], current_words: list[str]) -> int:
    """Uoc luong so word trung do overlap audio, uu tien match chinh xac."""
    max_words = min(30, len(previous_words), len(current_words))
    previous = [normalize_word(word) for word in previous_words]
    current = [normalize_word(word) for word in current_words]

    for size in range(max_words, 0, -1):
        left = previous[-size:]
        right = current[:size]
        if all(left) and left == right:
            return size

    # Model co the nhan sai mot vai tu o cung phan overlap. Chi fuzzy-merge khi
    # co it nhat 2 tu va do giong cao de tranh xoa nham noi dung moi.
    best_size = 0
    best_ratio = 0.0
    for size in range(2, max_words + 1):
        left = previous[-size:]
        right = current[:size]
        ratio = SequenceMatcher(None, left, right, autojunk=False).ratio()
        if ratio >= 0.75 and (ratio > best_ratio or (ratio == best_ratio and size > best_size)):
            best_size = size
            best_ratio = ratio
    return best_size


def merge_transcripts(results: Sequence[SegmentResult]) -> str:
    merged_words: list[str] = []
    for result in sorted(results, key=lambda item: item.index):
        words = result.text.split()
        if not words:
            continue
        if result.overlaps_previous and merged_words:
            repeated = number_of_repeated_prefix(merged_words, words)
            words = words[repeated:]
        merged_words.extend(words)
    return " ".join(merged_words).strip()


def is_url(value: str) -> bool:
    return urllib.parse.urlparse(value).scheme.lower() in {"http", "https"}


@contextmanager
def local_audio_source(source: str) -> Iterator[Path]:
    """Tra local path; URL duoc tai theo block vao file tam va xoa sau khi chay."""
    if not is_url(source):
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Khong tim thay audio: {path}")
        yield path
        return

    print(f"Dang tai audio: {source}")
    temporary_path: Path | None = None
    try:
        request = urllib.request.Request(source, headers={"User-Agent": "ASRProject/long-audio"})
        with urllib.request.urlopen(request) as response, tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                temporary_file.write(block)
        yield temporary_path
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def choose_device(device_name: str) -> torch.device:
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Da chon CUDA nhung torch.cuda.is_available() = False")
    if device.type != "cuda":
        raise RuntimeError(
            "ASR2026 hien chi chay duoc tren CUDA vi OptimizedFlashMHA chi bat "
            "SDPBackend.EFFICIENT_ATTENTION. Hay chay tren may co GPU CUDA."
        )
    return device


def audio_metadata(audio_path: Path) -> tuple[int, float]:
    with sf.SoundFile(audio_path) as audio_file:
        sample_rate = int(audio_file.samplerate)
        duration = audio_file.frames / sample_rate
    return sample_rate, duration


def validate_args(args: argparse.Namespace) -> None:
    if not 0 < args.chunk_seconds <= 15.0:
        raise ValueError("--chunk-seconds phai nam trong (0, 15]")
    if not 0 <= args.overlap_seconds < args.chunk_seconds:
        raise ValueError("--overlap-seconds phai >= 0 va nho hon --chunk-seconds")
    if not 0 <= args.silence_search_seconds < args.chunk_seconds:
        raise ValueError("--silence-search-seconds phai >= 0 va nho hon --chunk-seconds")
    if args.overlap_seconds + args.silence_search_seconds >= args.chunk_seconds:
        raise ValueError("Tong overlap va vung tim silence phai nho hon do dai cua so")
    if args.batch_size < 1:
        raise ValueError("--batch-size phai >= 1")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe WAV dai bang cac cua so silence-aware toi da 15 giay.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--audio",
        default=AUDIO_PATH_OR_URL,
        help="Duong dan local hoac URL HTTP/HTTPS cua WAV",
    )
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_TEXT_PATH, help="File .txt de luu transcript cuoi")
    parser.add_argument(
        "--segments-json",
        type=Path,
        default=SEGMENTS_JSON_PATH,
        help="File JSON chua timestamp va text tung doan",
    )
    parser.add_argument("--chunk-seconds", type=float, default=15.0, help="Do dai cua so toi da")
    parser.add_argument("--overlap-seconds", type=float, default=1.5, help="Overlap khi khong tim thay silence")
    parser.add_argument(
        "--silence-search-seconds",
        type=float,
        default=2.5,
        help="Vung tim diem im lang o cuoi moi cua so",
    )
    parser.add_argument(
        "--silence-threshold-db",
        type=float,
        default=-40.0,
        help="Frame thap hon nguong nay duoc xem la silence",
    )
    parser.add_argument(
        "--skip-silence-db",
        type=float,
        default=-65.0,
        help="Bo qua ca cua so neu RMS thap hon nguong nay; dat -inf de tat",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="So cua so inference cung luc")
    parser.add_argument("--beam-width", type=int, default=config.BEAM_WIDTH)
    parser.add_argument("--max-decode-length", type=int, default=256)
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0 hoac cpu")
    parser.add_argument("--no-amp", action="store_true", help="Tat float16 autocast tren CUDA")
    return parser


def save_results(args: argparse.Namespace, transcript: str, results: Sequence[SegmentResult]) -> None:
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(transcript + "\n", encoding="utf-8")
        print(f"Da luu transcript: {args.output.resolve()}")

    if args.segments_json:
        payload = {
            "audio": args.audio,
            "transcript": transcript,
            "segments": [
                {
                    "index": result.index,
                    "start": round(result.start_seconds, 3),
                    "end": round(result.end_seconds, 3),
                    "text": result.text,
                    "overlaps_previous": result.overlaps_previous,
                    "dbfs": round(result.dbfs, 2),
                    "score": result.score,
                    "skipped_as_silence": result.skipped_as_silence,
                }
                for result in sorted(results, key=lambda item: item.index)
            ],
        }
        args.segments_json.parent.mkdir(parents=True, exist_ok=True)
        args.segments_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Da luu segments: {args.segments_json.resolve()}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.audio:
        parser.error(
            "Hay dien AUDIO_PATH_OR_URL trong source hoac truyen --audio /duong/dan/audio.wav"
        )
    validate_args(args)

    checkpoint_path = args.checkpoint.expanduser().resolve()
    tokenizer_path = args.tokenizer.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Khong tim thay checkpoint: {checkpoint_path}")
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Khong tim thay tokenizer: {tokenizer_path}")

    device = choose_device(args.device)
    started_at = time.perf_counter()

    with local_audio_source(args.audio) as audio_path:
        source_rate, duration = audio_metadata(audio_path)
        print(f"Audio: {duration:.2f}s, {source_rate} Hz")

        transcriber = LongAudioTranscriber(
            checkpoint_path=checkpoint_path,
            tokenizer_path=tokenizer_path,
            device=device,
            beam_width=args.beam_width,
            max_decode_length=args.max_decode_length,
            use_amp=not args.no_amp,
        )

        pending: list[AudioChunk] = []
        results: list[SegmentResult] = []

        def flush_pending() -> None:
            if not pending:
                return
            batch_results = transcriber.transcribe_batch(pending, source_rate)
            results.extend(batch_results)
            for result in batch_results:
                print(
                    f"[{result.start_seconds:9.2f}s -> {result.end_seconds:9.2f}s] "
                    f"{result.text}"
                )
            pending.clear()

        for chunk in iter_audio_chunks(
            audio_path=audio_path,
            chunk_seconds=args.chunk_seconds,
            overlap_seconds=args.overlap_seconds,
            silence_search_seconds=args.silence_search_seconds,
            silence_threshold_db=args.silence_threshold_db,
        ):
            if chunk.dbfs < args.skip_silence_db:
                results.append(
                    SegmentResult(
                        index=chunk.index,
                        start_seconds=chunk.start_seconds,
                        end_seconds=chunk.end_seconds,
                        text="",
                        overlaps_previous=chunk.overlaps_previous,
                        dbfs=chunk.dbfs,
                        score=None,
                        skipped_as_silence=True,
                    )
                )
                print(
                    f"[{chunk.start_seconds:9.2f}s -> {chunk.end_seconds:9.2f}s] "
                    f"<silence {chunk.dbfs:.1f} dBFS>"
                )
                continue

            pending.append(chunk)
            if len(pending) >= args.batch_size:
                flush_pending()

        flush_pending()
        transcript = merge_transcripts(results)

    elapsed = time.perf_counter() - started_at
    print("\n===== TRANSCRIPT =====")
    print(transcript)
    print(f"\nHoan tat trong {elapsed:.2f}s")
    save_results(args, transcript, results)

if __name__ == "__main__":
    main()