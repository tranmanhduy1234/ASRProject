from transformers import LlamaTokenizerFast #type: ignore
from typing import List, Union

def find_closest_in_list(target):
    data_list = ['0.00', '0.05', '0.10', '0.15', '0.20', '0.25', '0.30', '0.35', '0.40', '0.45', '0.50', '0.55', '0.60', '0.65', '0.70', '0.75', '0.80', '0.85', '0.90', '0.95', '1.00', '1.05', '1.10', '1.15', '1.20', '1.25', '1.30', '1.35', '1.40', '1.45', '1.50', '1.55', '1.60', '1.65', '1.70', '1.75', '1.80', '1.85', '1.90', '1.95', '2.00', '2.05', '2.10', '2.15', '2.20', '2.25', '2.30', '2.35', '2.40', '2.45', '2.50', '2.55', '2.60', '2.65', '2.70', '2.75', '2.80', '2.85', '2.90', '2.95', '3.00', '3.05', '3.10', '3.15', '3.20', '3.25', '3.30', '3.35', '3.40', '3.45', '3.50', '3.55', '3.60', '3.65', '3.70', '3.75', '3.80', '3.85', '3.90', '3.95', '4.00', '4.05', '4.10', '4.15', '4.20', '4.25', '4.30', '4.35', '4.40', '4.45', '4.50', '4.55', '4.60', '4.65', '4.70', '4.75', '4.80', '4.85', '4.90', '4.95', 
                 '5.00', '5.05', '5.10', '5.15', '5.20', '5.25', '5.30', '5.35', '5.40', '5.45', '5.50', '5.55', '5.60', '5.65', '5.70', '5.75', '5.80', '5.85', '5.90', '5.95', '6.00', '6.05', '6.10', '6.15', '6.20', '6.25', '6.30', '6.35', '6.40', '6.45', '6.50', '6.55', '6.60', '6.65', '6.70', '6.75', '6.80', '6.85', '6.90', '6.95', '7.00', '7.05', '7.10', '7.15', '7.20', '7.25', '7.30', '7.35', '7.40', '7.45', '7.50', '7.55', '7.60', '7.65', '7.70', '7.75', '7.80', '7.85', '7.90', '7.95', '8.00', '8.05', '8.10', '8.15', '8.20', '8.25', '8.30', '8.35', '8.40', '8.45', '8.50', '8.55', '8.60', '8.65', '8.70', '8.75', '8.80', '8.85', '8.90', '8.95', '9.00', '9.05', '9.10', '9.15', '9.20', '9.25', '9.30', '9.35', '9.40', '9.45', '9.50', '9.55', '9.60', '9.65', '9.70', '9.75', '9.80', '9.85', '9.90', '9.95', 
                 '10.00', '10.05', '10.10', '10.15', '10.20', '10.25', '10.30', '10.35', '10.40', '10.45', '10.50', '10.55', '10.60', '10.65', '10.70', '10.75', '10.80', '10.85', '10.90', '10.95', '11.00', '11.05', '11.10', '11.15', '11.20', '11.25', '11.30', '11.35', '11.40', '11.45', '11.50', '11.55', '11.60', '11.65', '11.70', '11.75', '11.80', '11.85', '11.90', '11.95', '12.00', '12.05', '12.10', '12.15', '12.20', '12.25', '12.30', '12.35', '12.40', '12.45', '12.50', '12.55', '12.60', '12.65', '12.70', '12.75', '12.80', '12.85', '12.90', '12.95', '13.00', '13.05', '13.10', '13.15', '13.20', '13.25', '13.30', '13.35', '13.40', '13.45', '13.50', '13.55', '13.60', '13.65', '13.70', '13.75', '13.80', '13.85', '13.90', '13.95', '14.00', '14.05', '14.10', '14.15', '14.20', '14.25', '14.30', '14.35', '14.40', '14.45', '14.50', '14.55', '14.60', '14.65', '14.70', '14.75', '14.80', '14.85', '14.90', '14.95', 
                 '15.00', '15.05', '15.10', '15.15', '15.20', '15.25', '15.30', '15.35', '15.40', '15.45', '15.50', '15.55', '15.60', '15.65', '15.70', '15.75', '15.80', '15.85', '15.90', '15.95', '16.00', '16.05', '16.10', '16.15', '16.20', '16.25', '16.30', '16.35', '16.40', '16.45', '16.50', '16.55', '16.60', '16.65', '16.70', '16.75', '16.80', '16.85', '16.90', '16.95', '17.00', '17.05', '17.10', '17.15', '17.20', '17.25', '17.30', '17.35', '17.40', '17.45', '17.50', '17.55', '17.60', '17.65', '17.70', '17.75', '17.80', '17.85', '17.90', '17.95', '18.00', '18.05', '18.10', '18.15', '18.20', '18.25', '18.30', '18.35', '18.40', '18.45', '18.50', '18.55', '18.60', '18.65', '18.70', '18.75', '18.80', '18.85', '18.90', '18.95', '19.00', '19.05', '19.10', '19.15', '19.20', '19.25', '19.30', '19.35', '19.40', '19.45', '19.50', '19.55', '19.60', '19.65', '19.70', '19.75', '19.80', '19.85', '19.90', '19.95', '20.00']
    closest = min(data_list, key=lambda x: abs(float(x) - target))
    return closest

class Tokenizer2026:
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
    
    def decode(self, ids_batch: List[List[int]], skip_special_tokens: bool = True):
        if skip_special_tokens:
            ids_batch = self.remove_timestamp(ids_batch=ids_batch)
        return [
            self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
            for ids in ids_batch
        ]
    
    def remove_timestamp(self, ids_batch: List[List[int]]) -> List[List[int]]:
        return [
            [3 if 8 <= x <= 408 else x for x in row]
            for row in ids_batch
        ]
        
    def get_unk_token(self):
        return self.tokenizer.unk_token_id, self.tokenizer.unk_token
    def get_bos_token(self):
        return self.tokenizer.bos_token_id, self.tokenizer.bos_token
    def get_eos_token(self):
        return self.tokenizer.eos_token_id, self.tokenizer.eos_token 
    def get_pad_token(self):
        return self.tokenizer.pad_token_id, self.tokenizer.pad_token
    
    def get_timestamp_token(self, index):
        index = float(index)
        if float(index) >= 20.00:
            return 408, "<ts_20.00>"
        if float(index) <= 0.00:
            return 8, "<ts_0.00>"
        index = float(find_closest_in_list(index))
        return int(round(index / 0.05) + 8), f"<ts_{index:.2f}>"
    
    def get_vitranscript_token(self):
        return 4, "<|vitranscript|>"
    def get_entranslate_token(self):
        return 5, "<|entranslate|>"
    def get_startspeech_token(self):
        return 6, "<start_speech>"
    def get_endspeech_token(self):
        return 7, "<end_speech>"
        
if __name__ == "__main__":
    tokenizer = Tokenizer2026(
        model_spm_path=r'D:\chuyen_nganh\ASRProject\Tokenizer\unigram_10000.model',
        legacy=False
    )
    tokenizer.print_infor_vocab()
    batch_texts = ["trần đỗ mạnh duy " + tokenizer.get_timestamp_token(155.5)[1], tokenizer.get_timestamp_token(-150)[1], tokenizer.get_vitranscript_token()[1], 
                   tokenizer.get_entranslate_token()[1], tokenizer.get_startspeech_token()[1], tokenizer.get_endspeech_token()[1]]
    encoded, token_pieces = tokenizer.encode(texts=batch_texts)
    
    print(f"Batch encoding ({len(batch_texts)} texts):")
    print(f"Pieces: {token_pieces}")
    print(f"IDs: {encoded}")
    
    decoded = tokenizer.decode(encoded, skip_special_tokens=False)
    print(f"Decoded batch: {decoded}")