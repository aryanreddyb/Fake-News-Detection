import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
import os
import time

from dataset import RedditTextDataset

# Paths
TRAIN_PATH = r"d:\Capstone\fnd\data\processed\train.csv"
VAL_PATH = r"d:\Capstone\fnd\data\processed\validate.csv"
MODEL_SAVE_DIR = r"d:\Capstone\fnd\outputs\models\modernbert_baseline"

# Hyperparameters
MODEL_NAME = "answerdotai/ModernBERT-base"
BATCH_SIZE = 8          # Reduced for FP32 DeBERTa-v3
MAX_LENGTH = 128
EPOCHS = 2
LEARNING_RATE = 2e-5

def train():
    print("Setting up device...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cpu':
        print("WARNING: CUDA is not available. Training will be extremely slow.")

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)

    print("Loading datasets...")
    train_dataset = RedditTextDataset(TRAIN_PATH, tokenizer, MAX_LENGTH)
    val_dataset = RedditTextDataset(VAL_PATH, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)
    
    # Mixed Precision Scaler
    scaler = GradScaler(enabled=False)

    # Training Loop
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    best_val_f1 = 0.0

    print(f"Starting training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for step, batch in enumerate(train_loader):
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast(enabled=False):
                outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs.loss
                
            total_loss += loss.item()
            
            # Backward pass with scaler
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            if (step + 1) % 500 == 0:
                print(f"Epoch {epoch+1} | Step {step+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        print("Running validation...")
        model.eval()
        val_preds, val_labels = [], []
        
        for batch in val_loader:
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)
            
            with torch.no_grad():
                with autocast(enabled=False):
                    outputs = model(b_input_ids, attention_mask=b_input_mask)
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            val_preds.extend(preds)
            val_labels.extend(b_labels.cpu().numpy())
            
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Time: {time.time() - start_time:.2f}s")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            print("New best validation score! Saving model...")
            model.save_pretrained(MODEL_SAVE_DIR)
            tokenizer.save_pretrained(MODEL_SAVE_DIR)

    print("\nTraining complete! Best Validation F1:", best_val_f1)
    print("\nClassification Report (Validation):")
    print(classification_report(val_labels, val_preds))

if __name__ == "__main__":
    train()
