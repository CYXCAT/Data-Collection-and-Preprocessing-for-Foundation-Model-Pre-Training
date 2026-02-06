import torch
from data_collector import DataCollector
from data_cleaner import DataCleaner
from tokenizer_module import TokenizerModule
from data_loader import PretrainDataset, create_dataloader
from quality_analyzer import QualityAnalyzer
from config import Config

def save_sample_data(dataloader, num_batches=None):
    config = Config
    num_batches = num_batches or config.SAMPLE_BATCHES
    
    sample_batches = []
    for i, batch in enumerate(dataloader):
        if i < num_batches:
            sample_batches.append(batch)
        else:
            break
    
    filepath = config.OUTPUT_FILES["sample_data"]
    torch.save(sample_batches, filepath)
    print(f"Sample data saved: {filepath}, {len(sample_batches)} batches")
    print(f"Batch shape: {sample_batches[0]['input_ids'].shape}")

def main():
    config = Config
    
    print("=" * 60)
    print("Data Collection and Preprocessing Pipeline")
    print("=" * 60)
    
    print("\n[1/6] Collecting datasets...")
    collector = DataCollector(use_streaming=config.USE_STREAMING)
    raw_data = collector.collect_datasets()
    collector.save_raw_data(raw_data)
    
    print("\n[2/6] Cleaning data...")
    cleaner = DataCleaner()
    cleaned_data = cleaner.clean(raw_data)
    cleaner.save_cleaned_data(cleaned_data)
    
    print("\n[3/6] Tokenizing data...")
    tokenizer = TokenizerModule(multilingual=config.ENABLE_MULTILINGUAL)
    tokenized_data = tokenizer.tokenize(cleaned_data)
    tokenizer.save_tokenized_data(tokenized_data)
    
    print("\n[4/6] Creating data loader...")
    pad_token_id = tokenizer.tokenizer.pad_token_id
    dataset = PretrainDataset(tokenized_data, pad_token_id=pad_token_id)
    dataloader = create_dataloader(dataset)
    
    print("\n[5/6] Generating sample data...")
    save_sample_data(dataloader)
    
    if config.ENABLE_QUALITY_ANALYSIS:
        print("\n[6/6] Analyzing data quality...")
        analyzer = QualityAnalyzer()
        analyzer.analyze(tokenized_data)
    else:
        print("\n[6/6] Quality analysis skipped")
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
