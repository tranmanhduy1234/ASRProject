from jiwer import wer#type: ignore
import torch
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
import datetime
import warnings
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*") # đang sử dụng bản 80
import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import autocast, GradScaler
from config import * 
from Data.dataloader2026 import ASRDataloader
from Data.util import *
from Inference.beamsearch import BeamSearchOptim
from Tokenizer.tokenizer2025 import Tokenizer2025
from Model.build_component.model import ASR2026 
from Trainer.util import *
from Model.architecture.configmodel import EMBED_DIM
import torchaudio.transforms as T
import math

WRITER = None

def seed_everything(seed=28):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(SEED)

class WarmupLinearDecay:
    def __init__(self, warmup_steps, total_steps_update, base_lr=1e-4, max_lr=1e-3):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps_update
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.decay_steps = max(1, total_steps_update - warmup_steps)
    
    def get_lr(self, step):
        if step >= self.total_steps: 
            return self.base_lr
        if step < self.warmup_steps:
            return self.base_lr + (self.max_lr - self.base_lr) * step / self.warmup_steps
        else:
            progress = (step - self.warmup_steps) / self.decay_steps
            return self.max_lr * (1 - progress)
    def get_lrs(self):
        return [self.get_lr(step) for step in range(self.total_steps)]

def create_scheduler(optimizer, warmup_steps, total_steps_update, base_lr, max_lr):
    scheduler_config = WarmupLinearDecay(warmup_steps, total_steps_update, base_lr, max_lr)
    def lr_lambda(step):
        lr = scheduler_config.get_lr(step)
        return lr / scheduler_config.base_lr
    return LambdaLR(optimizer, lr_lambda)

def create_cosine_schedule_with_warmup(optimizer, num_warm_up, num_training_update_step, num_cycles, min_lr_ratio):
    def lr_lambda(current_step):
        if current_step < num_warm_up:
            return float(current_step) / float(max(1, num_warm_up))
        progress = float(current_step - num_warm_up) / float(max(1, num_training_update_step))
        progress = min(progress, 1.0)
        cosine_val = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_val
    return LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)

def get_noam_scheduler_warmup(optimizer, num_warmup_steps):
    def lr_lambda(current_step):
        current_step = max(1, current_step)
        arg1 = current_step ** (-0.5)
        arg2 = current_step * (num_warmup_steps ** (-1.5))
        lr_scale = min(arg1, arg2)
        return lr_scale * (EMBED_DIM ** (-0.5))
    return LambdaLR(optimizer, lr_lambda)

def validate_step(model: ASR2026, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation", leave=False)
        for batchdata in pbar:
            transcripts_target_preshift = batchdata['transcripts_target_preshift'].to(device)
            transcripts_target_shifted = batchdata['transcripts_target_shifted'].to(device)
            transcripts_mask = batchdata['transcripts_mask'].to(device)
            
            mel_spectrogram_dbs = batchdata['mel_spectrogram_dbs'].to(device)
            mel_mask = batchdata['mel_mask'].to(device)
            
            with autocast(device_type=device.type, enabled=True):
                output = model(mel_spectrogram_dbs, transcripts_target_preshift, mel_mask, transcripts_mask)
                loss = criterion(output.reshape(-1, output.shape[-1]), transcripts_target_shifted.reshape(-1))
            
            total_loss += loss.item()
            num_batches += 1
    avg_loss = total_loss / num_batches
    torch.cuda.empty_cache()
    return avg_loss

def train_ASR2026(model: ASR2026, train_loader, val_loader,
    optimizer, criterion, epochs,
    save_path, writer, beamsearchhead,
    scaler, scheduler, tokenizer,
    accumulation_steps, max_grad_norm, logging_step,
    save_step, device, total_step_training_per_epoch,
    rootfoldersave, last_epoch, last_step    
):
    model.train()
    smoothed_loss = 0.0
    
    for epoch in range(epochs):
        if epoch < last_epoch:
            continue
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=True, total=total_step_training_per_epoch)
        for idx, batchdata in enumerate(pbar):
            if idx <= last_step and epoch == last_epoch:
                pbar.update(1)
                continue
            transcripts_target_preshift = batchdata['transcripts_target_preshift'].to(device)
            transcripts_target_shifted = batchdata['transcripts_target_shifted'].to(device)
            transcripts_mask = batchdata['transcripts_mask'].to(device)
            
            mel_spectrogram_dbs = batchdata['mel_spectrogram_dbs'].to(device)
            mel_mask = batchdata['mel_mask'].to(device)
            
            with autocast(device_type=device.type, enabled=config.USE_AMP):
                output = model(mel_spectrogram_dbs, transcripts_target_preshift, mel_mask, transcripts_mask)
                loss = criterion(output.reshape(-1, output.shape[-1]), transcripts_target_shifted.reshape(-1))
            
            # Phòng vệ loss training
            if not torch.isfinite(loss):
                print(f"\n{'='*40}")
                print(f"CẢNH BÁO KHẨN CẤP: Loss bị hỏng ({loss.item()}) tại step {idx} - epoch: {epoch}")
                print(f"{'='*40}")
                
                print("THÔNG TIN BATCH GÂY LỖI:")
                print(f" - Kích thước Batch (Batch Size): {mel_spectrogram_dbs.shape[0]}")
                print(f" - Độ dài mel_spectrogram_dbs (Max Len En): {mel_spectrogram_dbs.shape[1]}")
                print(f" - Độ dài Target (Max Len Vi): {transcripts_target_shifted.shape[1]}")
                
                print(f"Vị trí lỗi - Epoch:{epoch} - Step: {idx}")
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                print("PHÁT HIỆN NaN LOSS ===> GRADIENT NaN")
                exit(0)
            # kết thúc
            
            loss = loss / accumulation_steps
            loss_value = loss.item() * accumulation_steps
            scaler.scale(loss).backward()
            
            if (idx + 1) % accumulation_steps == 0 or (idx + 1) == total_step_training_per_epoch:
                scaler.unscale_(optimizer)
                if max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                
                if (idx + 1) % logging_step == 0:
                    logGradient_histogram_mean_std(model=model, writer=writer, index=idx + epoch * total_step_training_per_epoch)
                
                optimizer.zero_grad(set_to_none=True)
                if old_scale <= scaler.get_scale():
                    scheduler.step()    
                current_lr = optimizer.param_groups[0]['lr']
                if (idx + 1) % logging_step == 0:
                    logLearningRate(writer=writer, lr=current_lr, step=idx + epoch * total_step_training_per_epoch)
                    
            if (idx + 1) % logging_step == 0:    
                logLoss(writer=writer,phase="Train" ,loss=loss_value, step=idx + epoch * total_step_training_per_epoch)
                logWeightBias_histogram_mean_std(model=model, writer=writer, index=idx + epoch * total_step_training_per_epoch)
                
            if (idx + 1) % save_step == 0 or (idx + 1) == total_step_training_per_epoch:
                save_checkpoint(model=model, 
                                optimizer=optimizer, 
                                scheduler=scheduler, 
                                scaler=scaler, 
                                step=idx,
                                epoch=epoch,
                                filepath=rootfoldersave + f"\checkpoint_{idx}_epoch_{epoch}.pt")
            if (idx + 1) % (save_step) == 0:
                model.eval()
                print("\nEvaluation...")
                loss_avg_val = validate_step(model=model, val_loader=val_loader, criterion=criterion, device=device)
                print(f"\nLoss validation: {loss_avg_val}")
                print("\n")
                logLoss(writer=writer, loss=loss_avg_val, phase="Validation", step=idx + epoch * total_step_training_per_epoch)
                model.train()
            smoothed_loss = smoothed_loss * 0.9 + loss_value * 0.1 if idx > 0 else loss_value
            pbar.set_postfix({'loss': f"{smoothed_loss:.4f}"})
    print(f"Hoàn thành training model trên {epochs} epochs")  
    print(f"Lưu model tại: {save_path}")
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        step=-1,
        epoch=-1,
        filepath=save_path,
    )
    print("Starting testing...")
    loss_avg_test = validate_step(model=model, val_loader=val_loader, criterion=criterion, device=device)
    print()
    print(f"Loss/Test: {loss_avg_test}")
    print("ENDING........................")
    
def WER_f(model: ASR2026, debug_loader, device, beamsearchhead: BeamSearchOptim, tokenizer: Tokenizer2025):
    model.eval()
    errors = 0
    total_samples = 0
    with torch.no_grad():
        pbar = tqdm(debug_loader, desc="WER debug", leave=False)
        for batchdata in pbar:
            transcripts = batchdata["transcripts"]
            ids, pieces = tokenizer.encode(transcripts)
            transcripts = tokenizer.decode(ids, skip_special_tokens=True)
            
            mel_spectrogram_dbs = batchdata['mel_spectrogram_dbs'].to(device) # torch([batch_size, 80, time])
            mel_mask = batchdata['mel_mask'].to(device)
            
            rs = None
            with autocast(device_type=device.type, enabled=True):
                rs, _ = beamsearchhead.batch_translate(audio_mel_spectrogram=mel_spectrogram_dbs, model=model, source_mask=mel_mask, use_cache=True)
                rs = rs.tolist()
            
            rs = tokenizer.decode(rs, skip_special_tokens=True)
            for index in range(len(rs)):
                error = wer(transcripts[index], rs[index])
                print(f"Tỷ lệ lỗi thành phần: {error}")
                if error > 1.0:
                    continue
                errors += error
                total_samples += 1
                
    torch.cuda.empty_cache()
    return errors / total_samples

class Trainer2026:
    def __init__(self):
        self.amplitute_to_db = T.AmplitudeToDB(top_db=100)
        self.mel_transform = T.MelSpectrogram(
            sample_rate=config.SAMPLE_RATE,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LEN,
            n_mels=config.CHANNEL_LOG_MEL
        )
        self.model = ASR2026().to(DEVICES)
        self.tokenizer2025 = Tokenizer2025(model_spm_path=MODEL_SPM_PATH, legacy=False)
        
        audio_address_database = load_database(config.ADDRESS_AUDIO)
        
        self.debug_dataloader = ASRDataloader(path_metadata=config.METADATA_DEBUG,
                                              tokenizer=self.tokenizer2025,
                                              mel_transform=self.mel_transform,
                                              amplitute_to_db=self.amplitute_to_db,
                                              audio_address_database=audio_address_database).getDataloader(batch_size=4)
        self.beamsearchhead = BeamSearchOptim(beam_width=BEAM_WIDTH, 
                                              max_len=MAX_LEN_INFERENCE, 
                                              sos_id=BOS, 
                                              eos_id=EOS,
                                              device=DEVICES)
        try: 
            load_checkpoint_onlymodel(config.LOAD_LAST_CHECKPOINT_PATH, self.model)
            wer = WER_f(self.model, debug_loader=self.debug_dataloader, 
                device=config.DEVICES, beamsearchhead=self.beamsearchhead, 
                tokenizer=self.tokenizer2025)
            print(f"Tỷ lệ lỗi: {wer * 100}%")
            exit(0)
        except Exception as e:
            print(e)
        self.train_dataloader = ASRDataloader(path_metadata=config.METADATA_TRAIN,
                                              tokenizer=self.tokenizer2025,
                                              mel_transform=self.mel_transform,
                                              amplitute_to_db=self.amplitute_to_db,
                                              audio_address_database=audio_address_database).getDataloader(batch_size=4)
        
        self.validation_dataloader = ASRDataloader(path_metadata=config.METADATA_TEST,
                                                   tokenizer=self.tokenizer2025,
                                                   mel_transform=self.mel_transform,
                                                   amplitute_to_db=self.amplitute_to_db,
                                                   audio_address_database=audio_address_database).getDataloader(batch_size=4)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=LEARNING_RATE,
            betas=BETAS,
            eps=EPS,
            weight_decay=WEIGHT_DECAY
        )
        self.criterion = torch.nn.CrossEntropyLoss(
            ignore_index=PAD,
            label_smoothing=SMOOTHING
        )
        self.scaler = GradScaler(enabled=config.USE_SCALER)
        self.scheduler = create_cosine_schedule_with_warmup(optimizer=self.optimizer,
                                                            num_warm_up=int((len(self.train_dataloader) // ACCUMULATION_STEPS + 1) * RATIO_WARMUP_GLOBAL_STEP * EPOCHS),
                                                            num_training_update_step=EPOCHS * (len(self.train_dataloader) // ACCUMULATION_STEPS + 1),
                                                            num_cycles=0.5,
                                                            min_lr_ratio=RATIO_DECAY_LEARNING_RATE)
        self.last_step, self.last_epoch = -2, -2
        
        if os.path.exists(LOAD_LAST_CHECKPOINT_PATH):
            self.last_step, self.last_epoch = load_checkpoint(
                filepath=LOAD_LAST_CHECKPOINT_PATH,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler
            )
        print(f"LastStep: {self.last_step} and LastEpoch: {self.last_epoch}")
        if self.last_epoch == -2:
            print("Train model from scratch starting...\n")

    def start_training(self):
        print(f"Starting training on {DEVICES}")
        print(f"Total training steps: {len(self.train_dataloader) * EPOCHS}")
        print(f"Total steps update: {EPOCHS * len(self.train_dataloader) // ACCUMULATION_STEPS}")
        print(f"Warmup steps update: {EPOCHS * int(len(self.train_dataloader) * RATIO_WARMUP_GLOBAL_STEP // ACCUMULATION_STEPS)}")
        
        train_ASR2026(
            model=self.model,
            train_loader=self.train_dataloader,
            val_loader=self.validation_dataloader,
            optimizer=self.optimizer,
            criterion=self.criterion,
            epochs=EPOCHS,
            save_path=SAVE_LAST_CHECKPOINT_PATH,
            writer=WRITER,
            beamsearchhead=self.beamsearchhead,
            scaler=self.scaler,
            scheduler=self.scheduler,
            tokenizer=self.tokenizer2025,
            accumulation_steps=ACCUMULATION_STEPS,
            max_grad_norm=MAX_GRAD_NORM,
            logging_step=LOGGING_STEP,
            save_step=SAVE_STEP,
            device=DEVICES,
            total_step_training_per_epoch=len(self.train_dataloader),
            rootfoldersave=ROOT_FOLDER_SAVE_CHECKPOINT,
            last_epoch=self.last_epoch,
            last_step=self.last_step
        )
        WRITER.close()
        print("Training complete")
if __name__=="__main__":
    WRITER = SummaryWriter(f'runs\{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
    trainer2026 = Trainer2026()
    trainer2026.start_training()