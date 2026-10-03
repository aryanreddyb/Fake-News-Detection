import pandas as pd
import torch
from torch.utils.data import Dataset

class RedditTextDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=128):
        """
        Args:
            csv_file (string): Path to the processed csv file.
            tokenizer (callable): HuggingFace tokenizer.
            max_length (int): Maximum length of the tokenized sequence.
        """
        # Read the CSV
        self.df = pd.read_csv(csv_file)
        
        # We ensure text is string and handle potential NaNs
        self.texts = self.df['clean_title'].fillna("").astype(str).tolist()
        self.labels = self.df['label'].astype(int).tolist()
        
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,      # Add '[CLS]' and '[SEP]'
            max_length=self.max_length,   # Pad & truncate
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',          # Return PyTorch tensors
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
