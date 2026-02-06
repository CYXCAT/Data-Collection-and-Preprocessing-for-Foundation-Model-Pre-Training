import os
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from config import Config

class DataCollector:
    def __init__(self, use_streaming=False):
        self.use_streaming = use_streaming
        self.config = Config
    
    def collect_datasets(self):
        if self.use_streaming:
            return self.collect_streaming()
        else:
            return self._collect_non_streaming()
    
    def _collect_non_streaming(self):
        all_dataframes = []
        
        for source_name, source_config in self.config.DATA_SOURCES.items():
            try:
                print(f"Loading {source_name}...")
                if source_name == "wikipedia":
                    dataset = load_dataset(
                        source_config["name"],
                        source_config["version"],
                        split="train",
                        streaming=False
                    )
                else:
                    dataset = load_dataset(
                        source_config["name"],
                        split="train",
                        streaming=False
                    )
                
                df = dataset.select_columns([source_config["text_field"]]).to_pandas()
                
                if source_config["text_field"] != "text":
                    df = df.rename(columns={source_config["text_field"]: "text"})
                
                if source_config.get("sample_frac"):
                    df = df.sample(frac=source_config["sample_frac"], random_state=42)
                
                all_dataframes.append(df)
                print(f"{source_name}: {len(df)} records")
            except Exception as e:
                print(f"Failed to load {source_name}: {e}")
                continue
        
        if not all_dataframes:
            raise ValueError("No datasets were successfully loaded")
        
        combined = pd.concat(all_dataframes, ignore_index=True)
        
        if "text" not in combined.columns:
            raise ValueError("Could not find 'text' column in combined dataset")
        
        return self._validate_and_adjust_size(combined)
    
    def collect_streaming(self):
        texts = []
        target_size_bytes = self.config.MIN_DATA_SIZE_GB * 1024 * 1024 * 1024
        current_size = 0
        
        for source_name, source_config in self.config.DATA_SOURCES.items():
            try:
                print(f"Streaming {source_name}...")
                if source_name == "wikipedia":
                    dataset = load_dataset(
                        source_config["name"],
                        source_config["version"],
                        split="train",
                        streaming=True
                    )
                else:
                    dataset = load_dataset(
                        source_config["name"],
                        split="train",
                        streaming=True
                    )
                
                for item in tqdm(dataset, desc=f"Collecting {source_name}"):
                    text = item.get(source_config["text_field"], "")
                    if text:
                        texts.append({"text": text})
                        current_size += len(text.encode('utf-8'))
                        if current_size >= target_size_bytes:
                            break
                
                if current_size >= target_size_bytes:
                    break
            except Exception as e:
                print(f"Failed to stream {source_name}: {e}")
                continue
        
        if not texts:
            raise ValueError("No data was collected")
        
        df = pd.DataFrame(texts)
        return self._validate_and_adjust_size(df)
    
    def _validate_and_adjust_size(self, df):
        temp_file = "temp_raw_data.csv"
        df.to_csv(temp_file, index=False)
        file_size_gb = os.path.getsize(temp_file) / (1024 ** 3)
        
        if file_size_gb < self.config.MIN_DATA_SIZE_GB:
            os.remove(temp_file)
            raise ValueError(f"Data size {file_size_gb:.2f} GB is less than required {self.config.MIN_DATA_SIZE_GB} GB")
        
        print(f"Collected data: {len(df)} records, {file_size_gb:.2f} GB")
        os.remove(temp_file)
        return df
    
    def save_raw_data(self, df, filepath=None):
        if filepath is None:
            filepath = self.config.OUTPUT_FILES["raw_data"]
        df.to_csv(filepath, index=False)
        file_size_gb = os.path.getsize(filepath) / (1024 ** 3)
        print(f"Raw data saved: {filepath}, {file_size_gb:.2f} GB")
