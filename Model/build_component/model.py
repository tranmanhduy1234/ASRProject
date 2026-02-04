import torch
import torch.nn as nn

from Model.build_component.decoderblock import DecoderBlock
from Model.build_component.encoderblock import EncoderBlock
from Model.build_component.embedding_decode import Embedding_Decode
from Model.build_component.preencode import AudioEncoderEmbedding
from Model.architecture import configmodel
import config

class ASR2026(nn.Module):
    def __init__(self):
        super().__init__()
        self.preEncode = AudioEncoderEmbedding(n_mels=config.CHANNEL_LOG_MEL, d_model=configmodel.EMBED_DIM)
        self.encode = nn.ModuleList([
            EncoderBlock(embed_dim=configmodel.EMBED_DIM, num_heads=configmodel.NUMHEAD,
                                                       ffn_hidden_dim=configmodel.D_FF, dropout=configmodel.ENCODE_DROPOUT[i], 
                                                       bias=configmodel.ENCODE_BIAS[i]) for i in range(configmodel.NUM_ENCODE)
            ])
        self.decode = nn.ModuleList([
            DecoderBlock(embed_dim=configmodel.EMBED_DIM, num_heads=configmodel.NUMHEAD, ffn_hidden_dim=configmodel.D_FF,
                        dropout=configmodel.DECODE_DROPOUT[i], bias=configmodel.DECODE_BIAS[i]) for i in range(configmodel.NUM_DECODE)
            ])
        self.embedding_decode = Embedding_Decode(vocab_size=configmodel.VOCAB, 
                                                 embed_dim=configmodel.EMBED_DIM, max_len=configmodel.MAXLEN, 
                                                 dropout=configmodel.EMBEDDING_DROPOUT)
        
        self.output_projection = nn.Linear(configmodel.EMBED_DIM, configmodel.VOCAB, bias=configmodel.OUTPUT_PROJ_BIAS)
        if self.output_projection.bias is not None:
            nn.init.constant_(self.output_projection.bias.data, 0.)
            
        self.output_projection.weight = self.embedding_decode.token_embed.weight
        self.src_kpmask_inference = None
        
    # src: torch.Size([4, 80, 1571]), src_kpmask: torch.Size([4, 1571])
    def forward(self, src, tgt, src_kpmask = None, tgt_kpmask=None):
        assert src.shape[1] == config.CHANNEL_LOG_MEL, "Cấu hình đầu vào không đúng"
        preEncoded = self.preEncode(src) # preEncoded: [batch_size, seqlen, d_model]
        batch_size, seq_len, _ = preEncoded.shape
        
        # src_mask: Torch([batch_size, mel_lengths])
        if src_kpmask is not None:
            mask_float = src_kpmask.float().unsqueeze(1) # Shape: [B, 1, Original_Len]
            # Resize mask về đúng seq_len của encoder
            src_key_padding_mask = torch.nn.functional.interpolate(
                mask_float, size=seq_len, mode='nearest'
            ).squeeze(1).bool()
        else:
            src_key_padding_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=src.device)
        
        # Lưu lại cho tiện dùng sau này
        self.src_key_padding_mask = src_key_padding_mask
        
        encoder_output = preEncoded
        for encoder_layer in self.encode:
            encoder_output = encoder_layer(encoder_output, key_padding_mask=src_key_padding_mask, is_causal=False)
        
        tgt_embedding = self.embedding_decode(tgt)
        decoder_output = tgt_embedding
        for decode_layer in self.decode:
            decoder_output = decode_layer(decoder_output, encoder_output,
                                        key_padding_mask_tgt = tgt_kpmask, 
                                        key_padding_mask_src = src_key_padding_mask,
                                        is_causal_self=True,
                                        is_causal_cross=False,
                                        use_cache=False
                                        )
        logits = self.output_projection(decoder_output)
        return logits
    
    def count_parameters(self):
        """Đếm tổng số parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_device(self):
        """Lấy device hiện tại của model"""
        return next(self.parameters()).device
    
    def reset_cache(self):
        for decoder_layer in self.decode:
            decoder_layer.reset_cache()
            
    def reorder_all_cache(self, beam_indices):
        for decoder_layer in self.decode:
            decoder_layer.reorder_cache(beam_indices)

    def inference_encoder(self, src, src_kpmask):
        # src_mask: Torch([batch_size, mel_lengths])
        assert src.shape[1] == config.CHANNEL_LOG_MEL
        preEncoded = self.preEncode(src) # preEncoded: [batch_size, seqlen, d_model]
        batch_size, seq_len, _ = preEncoded.shape
        if src_kpmask is not None:
            mask_float = src_kpmask.float().unsqueeze(1)
            src_kpmask = torch.nn.functional.interpolate(
                mask_float, size=seq_len, mode='nearest'
            ).squeeze(1).bool()
        else:
            src_kpmask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=src.device)
            
        encoder_output = preEncoded
        for encoder_layer in self.encode:
            encoder_output = encoder_layer(encoder_output, key_padding_mask=src_kpmask, is_causal=False)
            
        self.src_kpmask_inference = src_kpmask
        
        return encoder_output
    
    def inference_embedding_layer(self, input_embedding):
        return self.embedding_decode(input_embedding)

    def inference_decoder(self, tgt_embedding, encoder_output, src_mask=None, tgt_mask=None, 
                          is_causal_self=True, is_causal_cross=False, use_cache=False):
        decoder_output = tgt_embedding
        
        for decoder_layer in self.decode:
            decoder_output = decoder_layer(decoder_output, 
                                           encoder_output, 
                                           key_padding_mask_src = self.src_kpmask_inference,
                                           key_padding_mask_tgt = tgt_mask, 
                                           is_causal_self=is_causal_self, 
                                           is_causal_cross=is_causal_cross,
                                           use_cache=use_cache
                                           )
        return self.output_projection(decoder_output)

if __name__=="__main__":
    model = ASR2026().to("cuda")
    input_audio = torch.randn(8, 1571, 80).to("cuda").transpose(1, 2).contiguous()
    mel_mask = torch.ones((8, 1571), dtype=torch.bool).to("cuda")
    tgt_transcript = torch.randint(0, 10000, (8, 256)).to("cuda")
    tgt_mask = torch.ones((8, 256), dtype=torch.bool).to("cuda")
    
    print(model(input_audio, tgt_transcript, mel_mask, tgt_mask).shape)
    output_encoder = model.inference_encoder(input_audio, mel_mask)
    tgt_embedding = model.inference_embedding_layer(tgt_transcript).to("cuda")
    output_decoder = model.inference_decoder(tgt_embedding=tgt_embedding, encoder_output=output_encoder, src_mask=None, tgt_mask=tgt_mask)
    
    print(model.src_kpmask_inference.shape, "sss")
    print(output_encoder.shape)
    print(output_decoder.shape)
    print(model.count_parameters())
    print(model.get_device())
    model.reset_cache()