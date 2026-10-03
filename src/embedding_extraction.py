import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm

MODEL_NAME = "answerdotai/ModernBERT-base" # Or point to the baseline model directory if fine-tuned
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NODE_MAPPING_PATH = os.path.join(BASE_DIR, "outputs", "graphs", "node_mapping.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "embeddings")
MODEL_DIR = os.path.join(BASE_DIR, "outputs", "models", "modernbert_baseline")
BATCH_SIZE = 64

class NodeTextDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.texts = df['clean_title'].fillna("").astype(str).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

def extract_embeddings():
    print("Setting up device...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Load the fine-tuned weights if available, else fallback to base model
    model_to_load = MODEL_DIR if os.path.exists(MODEL_DIR) else MODEL_NAME
    model = AutoModelForSequenceClassification.from_pretrained(model_to_load)
    model.to(device)
    model.eval()

    print("Loading graph node mappings...")
    df = pd.read_csv(NODE_MAPPING_PATH)
    
    # Sort to ensure the embeddings align exactly with node_id (0 to N-1)
    df = df.sort_values(by="node_id").reset_index(drop=True)
    
    dataset = NodeTextDataset(df, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_embeddings = []

    print(f"Extracting embeddings for {len(df)} nodes...")
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Using half precision for faster extraction and lower memory
            with torch.autocast(device_type=device.type if device.type == 'cuda' else 'cpu'):
                outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            
            # Mean Pooling: Average all token embeddings, ignoring padding tokens
            token_embeddings = outputs.hidden_states[-1]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
            
            # Move to CPU immediately to prevent VRAM overflow
            all_embeddings.append(mean_embeddings.cpu())

    # Concatenate all batches into a single large tensor
    final_embeddings = torch.cat(all_embeddings, dim=0)
    print(f"Final Embedding Tensor Shape: {final_embeddings.shape}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, "node_embeddings.pt")
    torch.save(final_embeddings, save_path)
    print(f"Saved embeddings to {save_path}")

if __name__ == "__main__":
    extract_embeddings()
