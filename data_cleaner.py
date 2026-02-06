import re
import pandas as pd
from config import Config

class DataCleaner:
    def __init__(self):
        self.config = Config
    
    def clean(self, df):
        df = df.copy()
        
        df = self._remove_duplicates(df)
        df = self._normalize_text(df)
        df = self._remove_html_tags(df)
        df = self._filter_low_quality(df)
        
        return df[["cleaned_text"]].rename(columns={"cleaned_text": "text"})
    
    def _remove_duplicates(self, df):
        initial_count = len(df)
        df = df.drop_duplicates(subset="text", keep="first")
        removed = initial_count - len(df)
        print(f"Removed {removed} duplicate documents")
        return df.reset_index(drop=True)
    
    def _normalize_text(self, df):
        def normalize_text(text):
            text = str(text).lower()
            text = re.sub(r"\s+", " ", text)
            text = re.sub(r"[^\w\s\.\,\!\?]", "", text)
            return text.strip()
        
        df["cleaned_text"] = df["text"].apply(normalize_text)
        return df
    
    def _remove_html_tags(self, df):
        def remove_html_tags(text):
            return re.sub(r"<.*?>", "", text)
        
        df["cleaned_text"] = df["cleaned_text"].apply(remove_html_tags)
        return df
    
    def _filter_low_quality(self, df):
        initial_count = len(df)
        df["word_count"] = df["cleaned_text"].apply(lambda x: len(x.split()))
        df = df[df["word_count"] >= self.config.MIN_WORDS_PER_DOC].reset_index(drop=True)
        removed = initial_count - len(df)
        print(f"Removed {removed} low-quality documents (<{self.config.MIN_WORDS_PER_DOC} words)")
        return df
    
    def save_cleaned_data(self, df, filepath=None):
        if filepath is None:
            filepath = self.config.OUTPUT_FILES["cleaned_data"]
        df.to_csv(filepath, index=False)
        print(f"Cleaned data saved: {filepath}, {len(df)} records")
