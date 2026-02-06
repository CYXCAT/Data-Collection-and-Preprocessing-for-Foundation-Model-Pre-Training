import torch
import numpy as np
import matplotlib.pyplot as plt
from config import Config

class QualityAnalyzer:
    def __init__(self):
        self.config = Config
    
    def analyze(self, tokenized_chunks):
        self.analyze_token_distribution(tokenized_chunks)
        self.generate_statistics(tokenized_chunks)
    
    def analyze_token_distribution(self, tokenized_chunks):
        token_lengths = [len(chunk) for chunk in tokenized_chunks]
        
        plt.figure(figsize=(10, 6))
        plt.hist(token_lengths, bins=50, edgecolor='black')
        plt.title("Token Sequence Length Distribution")
        plt.xlabel("Sequence Length (tokens)")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        
        filepath = self.config.OUTPUT_FILES["length_distribution"]
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Length distribution plot saved: {filepath}")
    
    def generate_statistics(self, tokenized_chunks):
        token_lengths = [len(chunk) for chunk in tokenized_chunks]
        
        stats = {
            "total_chunks": len(tokenized_chunks),
            "mean_length": np.mean(token_lengths),
            "median_length": np.median(token_lengths),
            "std_length": np.std(token_lengths),
            "min_length": np.min(token_lengths),
            "max_length": np.max(token_lengths),
            "percentiles": {
                "25th": np.percentile(token_lengths, 25),
                "50th": np.percentile(token_lengths, 50),
                "75th": np.percentile(token_lengths, 75),
                "90th": np.percentile(token_lengths, 90),
                "95th": np.percentile(token_lengths, 95),
                "99th": np.percentile(token_lengths, 99)
            }
        }
        
        report_lines = [
            "Data Quality Analysis Report",
            "=" * 50,
            f"Total tokenized chunks: {stats['total_chunks']:,}",
            "",
            "Sequence Length Statistics:",
            f"  Mean: {stats['mean_length']:.2f} tokens",
            f"  Median: {stats['median_length']:.2f} tokens",
            f"  Std Dev: {stats['std_length']:.2f} tokens",
            f"  Min: {stats['min_length']} tokens",
            f"  Max: {stats['max_length']} tokens",
            "",
            "Percentiles:",
            f"  25th: {stats['percentiles']['25th']:.2f} tokens",
            f"  50th: {stats['percentiles']['50th']:.2f} tokens",
            f"  75th: {stats['percentiles']['75th']:.2f} tokens",
            f"  90th: {stats['percentiles']['90th']:.2f} tokens",
            f"  95th: {stats['percentiles']['95th']:.2f} tokens",
            f"  99th: {stats['percentiles']['99th']:.2f} tokens",
        ]
        
        report_text = "\n".join(report_lines)
        
        filepath = self.config.OUTPUT_FILES["quality_report"]
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print("\n" + report_text)
        print(f"\nQuality report saved: {filepath}")
