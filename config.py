import os

class Config:
    DATA_SOURCES = {
        "cc_news": {
            "name": "cc_news",
            "text_field": "text",
            "sample_frac": None
        }
    }
    
    MIN_DATA_SIZE_GB = 1.0
    MIN_WORDS_PER_DOC = 50
    MAX_SEQUENCE_LENGTH = 512
    MIN_CHUNK_LENGTH = 10
    
    TOKENIZER_MODEL = "gpt2"
    MULTILINGUAL_TOKENIZER = "bert-base-multilingual-uncased"
    
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    SAMPLE_BATCHES = 8
    
    OUTPUT_FILES = {
        "raw_data": "raw_data.csv",
        "cleaned_data": "cleaned_data.csv",
        "tokenized_data": "tokenized_data.pt",
        "sample_data": "sample_dataset.pt",
        "length_distribution": "length_distribution.png",
        "quality_report": "quality_report.txt"
    }
    
    USE_STREAMING = False
    ENABLE_MULTILINGUAL = False
    ENABLE_QUALITY_ANALYSIS = True
