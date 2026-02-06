import torch
from torch.utils.data import Dataset, DataLoader
from config import Config

class PretrainDataset(Dataset):
    def __init__(self, tokenized_chunks, max_block_size=None, pad_token_id=None):
        self.tokenized_chunks = tokenized_chunks
        self.max_block_size = max_block_size or Config.MAX_SEQUENCE_LENGTH
        self.pad_token_id = pad_token_id or 50256
    
    def __len__(self):
        return len(self.tokenized_chunks)
    
    def __getitem__(self, idx):
        chunk = self.tokenized_chunks[idx]
        
        if len(chunk) < self.max_block_size:
            padding = torch.full(
                (self.max_block_size - len(chunk),),
                self.pad_token_id,
                dtype=chunk.dtype
            )
            chunk = torch.cat([chunk, padding])
        elif len(chunk) > self.max_block_size:
            chunk = chunk[:self.max_block_size]
        
        return {"input_ids": chunk}

def create_dataloader(dataset, batch_size=None, shuffle=True, num_workers=None, pin_memory=True):
    config = Config
    batch_size = batch_size or config.BATCH_SIZE
    num_workers = num_workers or config.NUM_WORKERS
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return dataloader
