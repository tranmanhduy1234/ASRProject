import time
from Model.build_component.model import ASR2026
import torch
import torch.nn as nn
from Tokenizer.tokenizer2025 import Tokenizer2025 

class BeamSearchOptim(nn.Module):
    def __init__(self, beam_width, max_len, sos_id, eos_id, device='cuda', alpha=0.6, per_beam_k=None):
        super().__init__()
        self.B = beam_width
        self.max_len = max_len
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.device = device
        self.alpha = alpha
        self.per_beam_k = per_beam_k
    
    @torch.no_grad()
    def batch_translate(self, audio_mel_spectrogram, model: ASR2026, source_mask=None, use_cache=False):
        # audio_mel_spectrogram = torch([batch_size, 80, time])
        batch_size, _, _ = audio_mel_spectrogram.shape
        encoder_output = model.inference_encoder(src=audio_mel_spectrogram, src_kpmask=source_mask)
        # [batch_size, seq_len_src, embed_dim]
        
        encoder_output = encoder_output.unsqueeze(1).expand(-1, self.B, -1, -1).contiguous()
        # [batch_size, beam_width, seq_len_src, embed_dim]
        model.src_kpmask_inference = model.src_kpmask_inference.unsqueeze(1).expand(-1, self.B, -1).contiguous()
        model.src_kpmask_inference = model.src_kpmask_inference.reshape(batch_size * self.B, -1)
        
        encoder_output = encoder_output.reshape(batch_size * self.B, -1, encoder_output.shape[-1])
        # [batch_size * beam_width, seq_len_src, embed_dim]
        
        # Xử lý source_mask nếu không None
        if source_mask is not None:
            source_mask = source_mask.unsqueeze(1).expand(-1, self.B, -1).contiguous()
            # [batch_size, beam_width, seq_len_src]
            source_mask = source_mask.reshape(batch_size * self.B, -1) 
            # [batch_size* beam_width, seq_len_src]

        beam_seqs = torch.full((batch_size * self.B, 1), self.sos_id, dtype=torch.long, device=self.device)
        beam_scores = torch.zeros((batch_size, self.B), device=self.device)
        finished = torch.zeros((batch_size, self.B), dtype=torch.bool, device=self.device)
        beam_lengths = torch.ones((batch_size, self.B), dtype=torch.long, device=self.device)

        model.reset_cache()
        for step in range(self.max_len):
            beam_seqs_last_token = None
            if use_cache:
                beam_seqs_last_token = beam_seqs[:, -1:]
            
            beam_seqs_embed = model.inference_embedding_layer(beam_seqs_last_token) if use_cache else model.inference_embedding_layer(beam_seqs)
            # [batch_size * beam_width, seq_len_query, embed_dim]
            logits = model.inference_decoder(
                tgt_embedding=beam_seqs_embed, # truyền embedding token của phần tử cuối mỗi beam/batch
                encoder_output=encoder_output, 
                tgt_mask=None, 
                src_mask=source_mask,
                is_causal_self=True if step == 0 else False,   
                is_causal_cross=False,
                use_cache=use_cache
            )
            # [batch_size * beam_width, seq_len_query, vocab_size]

            next_token_logits = logits[:, -1, :]
            # [batch_size * beam_width, vocab_size]
            
            log_probs = torch.log_softmax(next_token_logits, dim=-1)
            # [batch_size * beam_width, vocab_size]
            
            # Reshape để xử lý per-batch
            log_probs = log_probs.view(batch_size, self.B, -1)
            # [batch_size, beam_width, vocab_size]
            
            # Prevent expansion of finished beams
            if finished.any():
                log_probs = log_probs.clone()
                log_probs[finished, :] = -float("inf")
                log_probs[finished, self.eos_id] = 0.0

            vocab_size = log_probs.shape[-1]
            k = self.per_beam_k or min(vocab_size, self.B * 4)
            
            # Per-beam topk
            topk_vals, topk_ids = torch.topk(log_probs, k, dim=-1)
            # [batch_size, beam_width, k]
            
            # Compute candidate scores
            cand_scores = beam_scores.unsqueeze(2) + topk_vals
            # [batch_size, beam_width, k]
            
            # Flatten và lấy top B candidates cho mỗi batch
            flat_scores = cand_scores.view(batch_size, -1)
            # [batch_size, beam_width * k]
            
            topk_flat_scores, topk_flat_indices = torch.topk(flat_scores, self.B, dim=-1)
            # [batch_size, beam_width]
            
            # Tìm parent beam và token position
            parent_beam_indices = topk_flat_indices // k
            # [batch_size, beam_width]
            
            chosen_token_positions = topk_flat_indices % k
            # [batch_size, beam_width]
            
            # Lấy token IDs từ topk_ids
            batch_indices = torch.arange(batch_size, device=self.device).view(batch_size, 1).expand(batch_size, self.B)
            
            chosen_token_indices = topk_ids[batch_indices, parent_beam_indices, chosen_token_positions]
            # [batch_size, beam_width]
            
            # Update beam sequences
            offsets = torch.arange(batch_size, device=self.device).view(batch_size, 1) * self.B
            global_parent_indices = parent_beam_indices + offsets
            # [batch_size, beam_width]
            
            global_parent_indices_flat = global_parent_indices.view(-1)
            if use_cache:
                model.reorder_all_cache(global_parent_indices_flat)
            
            beam_seqs = torch.cat([beam_seqs[global_parent_indices_flat], chosen_token_indices.view(-1, 1)], dim=-1)
            # [batch_size * beam_width, seq_len_tgt + 1]
            
            # Update finished status
            is_eos = (chosen_token_indices == self.eos_id)
            finished = finished[batch_indices, parent_beam_indices] | is_eos
            beam_lengths = beam_lengths[batch_indices, parent_beam_indices].clone()
            beam_lengths[~finished] += 1
            
            # Update scores
            beam_scores = topk_flat_scores 
            # [batch_size * beam_width]
            
            if finished.all():
                break

        # Apply length penalty
        float_lengths = beam_lengths.float()
        length_penalty = torch.pow((5.0 + float_lengths) / 6.0, self.alpha)
        final_scores = beam_scores / length_penalty
        
        # Lấy best sequence cho mỗi batch
        best_indices = torch.argmax(final_scores, dim=-1)
        # [batch_size]
        
        offsets = torch.arange(batch_size, device=self.device) * self.B
        global_best_indices = best_indices + offsets
        
        return beam_seqs[global_best_indices], final_scores[torch.arange(batch_size, device=self.device), best_indices]
        
if __name__=="__main__":
    use_cache = True
    audio_mel_spectrogram = torch.randn(16, 80, 2000).to("cuda")
    source_mask = torch.randn(16, 2000).to("cuda")
    model = ASR2026().to('cuda')
    model.eval()
    import time
    start = time.time()
    
    with torch.no_grad():
        # dau vao: torch([batch_size, 80, time])
        beamsearchhead = BeamSearchOptim(beam_width=5, max_len=256, sos_id=1, eos_id=2, device='cuda', alpha=0.6)
        rs, _ = beamsearchhead.batch_translate(audio_mel_spectrogram=audio_mel_spectrogram, model=model, source_mask=source_mask, use_cache=use_cache)
        print(rs, "\n", rs.shape)
    print(time.time() - start)
    rs = rs.tolist()
    tokenizer2025 = Tokenizer2025(model_spm_path=r"D:\chuyen_nganh\ASRProject\Tokenizer\unigram_10000.model", legacy=False)
    print(tokenizer2025.decode(rs, skip_special_tokens=True))