import torch
from transformers import AutoTokenizer
from tqdm import tqdm
from config import Config

class TokenizerModule:
    def __init__(self, model_name=None, multilingual=False):
        self.config = Config
        self.multilingual = multilingual
        
        if multilingual:
            model_name = model_name or self.config.MULTILINGUAL_TOKENIZER
        else:
            model_name = model_name or self.config.TOKENIZER_MODEL
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.max_block_size = self.config.MAX_SEQUENCE_LENGTH
        self.min_chunk_length = self.config.MIN_CHUNK_LENGTH
    
    def tokenize(self, df):
        if self.multilingual:
            return self.tokenize_multilingual(df)
        
        texts = df["text"].tolist()
        all_chunks = []
        
        for text in tqdm(texts, desc="Tokenizing"):
            chunks = self._tokenize_text(text)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def tokenize_multilingual(self, df):
        texts = df["text"].tolist()
        all_chunks = []
        
        for text in tqdm(texts, desc="Tokenizing (multilingual)"):
            chunks = self._tokenize_text(text)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def _tokenize_text(self, text):
        tokens = self.tokenizer(
            text,
            truncation=False,
            padding=False,
            return_tensors="pt"
        )
        
        input_ids = tokens["input_ids"].flatten()
        chunks = [
            input_ids[i:i+self.max_block_size]
            for i in range(0, len(input_ids), self.max_block_size)
        ]
        
        chunks = [chunk for chunk in chunks if len(chunk) >= self.min_chunk_length]
        return chunks
    
    def batch_tokenize(self, texts, batch_size=32):
        all_chunks = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Batch tokenizing"):
            batch_texts = texts[i:i+batch_size]
            for text in batch_texts:
                chunks = self._tokenize_text(text)
                all_chunks.extend(chunks)
        
        return all_chunks
    
    def save_tokenized_data(self, tokenized_chunks, filepath=None):
        if filepath is None:
            filepath = self.config.OUTPUT_FILES["tokenized_data"]
        torch.save(tokenized_chunks, filepath)
        print(f"Tokenized data saved: {filepath}, {len(tokenized_chunks)} chunks")
