import sentencepiece as spm
from typing import List, Union
import numpy as np

class Tokenizer2025:
    def __init__(self, model_spm_path="", legacy=False):
        self.model_spm_path = model_spm_path
        self.sp = spm.SentencePieceProcessor()
        # Load model và kiểm tra
        if not self.sp.load(model_spm_path):
            raise IndexError(f"Không thể load model từ: {model_spm_path}")
        
        self.pad_token = "<pad>"
        # piece_to_id trả về ID hoặc unk_id nếu không tìm thấy
        self.pad_id = self.sp.piece_to_id(self.pad_token)
        if self.pad_id == self.sp.unk_id() and self.pad_token not in self.sp:
             self.pad_id = -1 # Đánh dấu nếu thực sự không có trong vocab
        self.print_infor_vocab()
    def print_infor_vocab(self):
        print(f"Kích thước từ điển: {self.sp.get_piece_size()}")
        print(f"UNK token: {self.sp.id_to_piece(self.sp.unk_id())} (ID: {self.sp.unk_id()})")
        print(f"BOS token: {self.sp.id_to_piece(self.sp.bos_id())} (ID: {self.sp.bos_id()})")
        print(f"EOS token: {self.sp.id_to_piece(self.sp.eos_id())} (ID: {self.sp.eos_id()})")
        print(f"PAD token: {self.pad_token} (ID: {self.pad_id})")
        
    def encode(self, texts: List[str]):
        all_ids = []
        all_pieces = []
        
        bos_id = self.sp.bos_id()
        eos_id = self.sp.eos_id()

        for text in texts:
            # Lỗi "unknown output" thường do truyền nhầm type vào encode
            # Đảm bảo text là string và out_type là int
            ids = self.sp.encode(str(text), out_type=int)
            
            # Ghép thủ công BOS/EOS
            full_ids = [bos_id] + ids + [eos_id]
            
            all_ids.append(np.array(full_ids))
            # Fix: id_to_piece chỉ nhận 1 ID (int), không nhận list/array trực tiếp
            all_pieces.append([self.sp.id_to_piece(int(idx)) for idx in full_ids])

        return all_ids, all_pieces
    
    def decode(self, ids_batch: List[Union[List[int], np.ndarray]], skip_special_tokens: bool = True):
        decoded_texts = []
        for ids in ids_batch:
            # Chuyển numpy array về list int chuẩn
            if isinstance(ids, np.ndarray):
                curr_ids = ids.tolist()
            else:
                curr_ids = list(ids)
            
            # Đảm bảo mọi phần tử là int (tránh lỗi input type)
            curr_ids = [int(x) for x in curr_ids]
            
            if skip_special_tokens:
                special_ids = {self.sp.bos_id(), self.sp.eos_id(), self.sp.unk_id(), self.pad_id}
                curr_ids = [idx for idx in curr_ids if idx not in special_ids]
            
            # sp.decode nhận list of ints
            decoded_texts.append(self.sp.decode(curr_ids))
        return decoded_texts

    def get_unk_token(self): return self.sp.unk_id(), self.sp.id_to_piece(self.sp.unk_id())
    def get_bos_token(self): return self.sp.bos_id(), self.sp.id_to_piece(self.sp.bos_id())
    def get_eos_token(self): return self.sp.eos_id(), self.sp.id_to_piece(self.sp.eos_id())
    def get_pad_token(self): return self.pad_id, self.pad_token

if __name__ == "__main__":
    MODEL_PATH = r'D:\chuyen_nganh\Machine-Translation\source\tokenizer\unigram_40000.model'
    
    tokenizer = Tokenizer2025(model_spm_path=MODEL_PATH)
    tokenizer.print_infor_vocab()
    
    batch_texts = ["Trần Đỗ Mạnh Duy. Ngày mai rồi sẽ khác", "Xin chào thế giới"]
    encoded_ids, token_pieces = tokenizer.encode(texts=batch_texts)
    
    print(f"\nCâu 1 IDs: {encoded_ids[0]}")
    # Decode thử nghiệm
    decoded = tokenizer.decode([encoded_ids[0]])
    print(f"Decoded: {decoded}")