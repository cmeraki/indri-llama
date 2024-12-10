import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import pickle
import os
from sklearn.model_selection import train_test_split
from llama_model import Llama, LlamaConfig
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

class TokenDataset(data.Dataset):
    def __init__(self, tokens):
        self.tokens = tokens

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        seq = torch.tensor(self.tokens[idx].clone().detach(), dtype=torch.int32)
        
        task_token_idx = torch.where(seq == 144642)[0]
        speaker_token_idx = torch.where(seq == 144645)[0]
        audio_start_token_idx = torch.where(seq == 144641)[0]
        stop_token_idx = torch.where(seq == 144644)[0]
        
        if (len(task_token_idx) == 0 or len(speaker_token_idx) == 0 or 
            len(audio_start_token_idx) == 0 or len(stop_token_idx) == 0):
            raise ValueError("Required special tokens not found in sequence")
        
        task_token_idx = task_token_idx[0].item()
        speaker_token_idx = speaker_token_idx[0].item()
        audio_start_token_idx = audio_start_token_idx[0].item()
        stop_token_idx = stop_token_idx[0].item()
        
        inputs = seq[:audio_start_token_idx + 1]        
        targets = seq[audio_start_token_idx + 1:stop_token_idx + 1]
        
        return inputs, targets

def pad_collate_fn(batch, max_input_len=2048, max_target_len=2048):
    inputs, targets = zip(*batch)

    padded_inputs = torch.full((len(inputs), max_input_len), fill_value=0, dtype=torch.int32)
    input_masks = torch.zeros((len(inputs), max_input_len), dtype=torch.bool)
    
    for i, seq in enumerate(inputs):
        seq_len = min(len(seq), max_input_len)
        padded_inputs[i, :seq_len] = seq[:seq_len]
        input_masks[i, :seq_len] = 1

    padded_targets = torch.full((len(targets), max_target_len), fill_value=0, dtype=torch.int32)
    target_masks = torch.zeros((len(targets), max_target_len), dtype=torch.bool)
    
    for i, seq in enumerate(targets):
        seq_len = min(len(seq), max_target_len)
        padded_targets[i, :seq_len] = seq[:seq_len]
        target_masks[i, :seq_len] = 1

    return padded_inputs, padded_targets, input_masks, target_masks

def train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=10, scheduler=None):
    best_val_loss = float('inf')
    accumulation_steps = 4  
    scaler = torch.amp.GradScaler(device)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        optimizer.zero_grad()

        for batch_idx, (inputs, targets, input_masks, target_masks) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            input_masks = input_masks.to(device)
            target_masks = target_masks.to(device)

            with torch.amp.autocast(device):
                outputs = model(inputs, attention_mask=input_masks)
                
                loss = criterion(
                    outputs.view(-1, outputs.size(-1))[target_masks.view(-1)], 
                    targets.view(-1)[target_masks.view(-1)]
                ) / accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if scheduler:
                    scheduler.step()

            total_train_loss += loss.item() * accumulation_steps

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets, input_masks, target_masks in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                input_masks = input_masks.to(device)
                target_masks = target_masks.to(device)

                with torch.amp.autocast(device):
                    outputs = model(inputs, attention_mask=input_masks)
                    val_loss = criterion(
                        outputs.view(-1, outputs.size(-1))[target_masks.view(-1)], 
                        targets.view(-1)[target_masks.view(-1)]
                    )
                
                total_val_loss += val_loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss
            }, 'best_model_checkpoint.pth')

    return model

def main():
    seed_everything(42)
    torch.backends.cudnn.enabled = False
    
    tokens = load_data('tokens/lj_speech_tokens.pkl')
    
    print(f"Total tokens: {len(tokens)}")
    print(f"Sample token shape: {len(tokens[0])}")
    print(f"Token value range: {min(min(seq) for seq in tokens)}-{max(max(seq) for seq in tokens)}")
    
    train_tokens, test_tokens = train_test_split(tokens, test_size=0.2, random_state=42)
    val_tokens, test_tokens = train_test_split(test_tokens, test_size=0.5, random_state=42)

    train_dataset = TokenDataset(train_tokens)
    val_dataset = TokenDataset(val_tokens)
    test_dataset = TokenDataset(test_tokens)

    train_loader = data.DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True, 
        collate_fn=lambda batch: pad_collate_fn(batch, max_input_len=512, max_target_len=512),
        num_workers=4,  
        pin_memory=True  
    )
    val_loader = data.DataLoader(
        val_dataset, 
        batch_size=32, 
        shuffle=False, 
        collate_fn=lambda batch: pad_collate_fn(batch, max_input_len=512, max_target_len=512),
        num_workers=4,
        pin_memory=True
    )
    test_loader = data.DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False, 
        collate_fn=lambda batch: pad_collate_fn(batch, max_input_len=512, max_target_len=512),
        num_workers=4,
        pin_memory=True
    )

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    config = LlamaConfig(
        vocab_size=max(max(seq) for seq in tokens) + 1,  
        dim=2048, 
        n_layers=12, 
        n_heads=16
    )
    
    model = Llama(config).to(device)
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=1e-4, 
        weight_decay=0.01  
    )
    
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=len(train_loader) * 10,  
        eta_min=1e-6 
    )
    
    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.1,  
        ignore_index=0  
    )

    trained_model = train_model(
        model, 
        train_loader, 
        val_loader, 
        optimizer, 
        criterion, 
        device, 
        num_epochs=10,
        scheduler=scheduler
    )

    torch.save(trained_model.state_dict(), 'final_llama_model.pth')

if __name__ == '__main__':
    main()