from transformers import LlamaTokenizerFast #type: ignore
from typing import List, Union

class Tokenizer2025:
    def __init__(self, model_spm_path="", legacy=False):
        self.model_spm_path = model_spm_path
        self.legacy = legacy
        self.tokenizer = LlamaTokenizerFast(vocab_file=self.model_spm_path, legacy=False)
        self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
        
    def print_infor_vocab(self):
        print(f"Kích thước từ điển: {len(self.tokenizer)}")
        print(f"UNK token: {self.tokenizer.unk_token} (ID: {self.tokenizer.unk_token_id})")
        print(f"BOS token: {self.tokenizer.bos_token} (ID: {self.tokenizer.bos_token_id})")
        print(f"EOS token: {self.tokenizer.eos_token} (ID: {self.tokenizer.eos_token_id})")
        print(f"PAD token: {self.tokenizer.pad_token} (ID: {self.tokenizer.pad_token_id})")
        
    def encode(self, texts: Union[List[str]]):
        texts = ["".join([str(self.tokenizer.bos_token), text, str(self.tokenizer.eos_token)]) 
                for text in texts
            ]
        encoded = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=texts,
            return_tensors='np',
            add_special_tokens=False
        )
        token_ids = encoded['input_ids']
        token_pieces = [
            self.tokenizer.convert_ids_to_tokens(ids) 
            for ids in token_ids
        ]
        return token_ids, token_pieces
    
    def decode(self, ids_batch, skip_special_tokens: bool = True):
        return [
            self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
            for ids in ids_batch
        ]
    
    def get_unk_token(self):
        return self.tokenizer.unk_token_id, self.tokenizer.unk_token
    def get_bos_token(self):
        return self.tokenizer.bos_token_id, self.tokenizer.bos_token
    def get_eos_token(self):
        return self.tokenizer.eos_token_id, self.tokenizer.eos_token 
    def get_pad_token(self):
        return self.tokenizer.pad_token_id, self.tokenizer.pad_token
            
if __name__ == "__main__":
    tokenizer = Tokenizer2025(
        model_spm_path=r'D:\chuyen_nganh\ASRProject\Tokenizer\unigram_10000.model',
        legacy=False
    )
    tokenizer.print_infor_vocab()
    
    batch_texts = ["trần đỗ mạnh duy. ngày mai rồi sẽ khác", "xin chào thế giới tối ưu hóa tốc độ"]
    encoded, token_pieces = tokenizer.encode(texts=batch_texts)
    decoded = tokenizer.decode(encoded)
    print(encoded)
    print(decoded)