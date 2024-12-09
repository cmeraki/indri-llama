import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
import numpy as np
from llama_model import get_model  

class TrainingConfig:
    def __init__(
        self, 
        batch_size=32, 
        grad_accumulation_steps=4,
        learning_rate=1e-4, 
        weight_decay=0.01,
        epochs=1000,
        max_grad_norm=1.0,
        warmup_steps=100,
        log_interval=100,
        checkpoint_path='Llama_tokenizer/model.safetensors'
    ):
        self.batch_size = batch_size
        self.grad_accumulation_steps = grad_accumulation_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.max_grad_norm = max_grad_norm
        self.warmup_steps = warmup_steps
        self.log_interval = log_interval
        self.checkpoint_path = checkpoint_path

class AudioTokenDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def prepare_data(pkl_path, test_split=0.1, val_split=0.1):
    with open(pkl_path, 'rb') as f:
        all_tokens = pickle.load(f)
    
    if not isinstance(all_tokens, torch.Tensor):
        all_tokens = torch.tensor(all_tokens, dtype=torch.long)
    
    total_size = len(all_tokens)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - test_size - val_size
    
    train_data, val_data, test_data = random_split(
        all_tokens, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    return {
        'train': AudioTokenDataset(train_data),
        'val': AudioTokenDataset(val_data),
        'test': AudioTokenDataset(test_data)
    }

def create_batch_collate(start_token=128000, audio_start_token=144461, stop_token=144644):
    def collate_fn(batch):
        def find_special_token_indices(sequence):
            try:
                audio_start_idx = torch.where(sequence == audio_start_token)[0][0]
                stop_idx = torch.where(sequence == stop_token)[0][0]
                return audio_start_idx, stop_idx
            except IndexError:
                print("Warning: Special tokens not found in sequence")
                return None, None
        
        processed_sequences = []
        targets = []
        
        for sequence in batch:
            audio_start_idx, stop_idx = find_special_token_indices(sequence)
            
            if audio_start_idx is None or stop_idx is None:
                continue
            
            input_seq = sequence[:audio_start_idx+1]
            
            target_seq = sequence[audio_start_idx+1:stop_idx]
            
            processed_sequences.append(input_seq)
            targets.append(target_seq)
        
        input_seq_padded = torch.nn.utils.rnn.pad_sequence(
            processed_sequences, 
            batch_first=True, 
            padding_value=0  
        )
        
        target_seq_padded = torch.nn.utils.rnn.pad_sequence(
            targets, 
            batch_first=True, 
            padding_value=-100 
        )
        
        return input_seq_padded, target_seq_padded
    
    return collate_fn

def create_learning_rate_scheduler(optimizer, config):
    """
    Create a learning rate scheduler with warmup
    """
    def lr_lambda(current_step: int):
        if current_step < config.warmup_steps:
            return float(current_step) / float(max(1, config.warmup_steps))
        return 1.0  
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_model(
    model, 
    dataloaders, 
    config: TrainingConfig, 
    device
):
    
    best_val_loss = float('inf')
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay
    )
    
    lr_scheduler = create_learning_rate_scheduler(optimizer, config)
    
    global_step = 0
    
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        for batch_idx, (input_features, targets) in enumerate(dataloaders['train']):
            input_features = input_features.to(device)
            targets = targets.to(device)
            
            loss = model.forward_loss(input_features, targets)
            
            loss = loss / config.grad_accumulation_steps
            
            loss.backward()
            
            if (batch_idx + 1) % config.grad_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
            
            train_loss += loss.item()
            
            if batch_idx % config.log_interval == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Train Loss: {loss.item():.4f}')
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for input_features, targets in dataloaders['val']:
                input_features = input_features.to(device)
                targets = targets.to(device)
                
                loss = model.forward_loss(input_features, targets)
                val_loss += loss.item()
        
        train_loss /= len(dataloaders['train'])
        val_loss /= len(dataloaders['val'])
        
        print(f'Epoch {epoch}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }, config.checkpoint_path)
        
    return model

def main():
    config = TrainingConfig(
        batch_size=2,  
        grad_accumulation_steps=16,
        learning_rate=1e-4,
        epochs=1000
    )
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = get_model(
        model_type='llama',
        path='Llama_tokenizer/model.safetensors',
        device=str(DEVICE),
        audio_feature_dim=128  
    )
    model.to(DEVICE)
    
    datasets = prepare_data('tokens/lj_speech_tokens.pkl')
    
    dataloaders = {
        'train': DataLoader(
            datasets['train'], 
            batch_size=config.batch_size, 
            shuffle=True, 
            collate_fn=create_batch_collate()
        ),
        'val': DataLoader(
            datasets['val'], 
            batch_size=config.batch_size, 
            shuffle=False, 
            collate_fn=create_batch_collate()
        ),
        'test': DataLoader(
            datasets['test'], 
            batch_size=config.batch_size, 
            shuffle=False, 
            collate_fn=create_batch_collate()
        )
    }
    
    trained_model = train_model(
        model, 
        dataloaders, 
        config, 
        DEVICE
    )
    
    trained_model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for input_features, targets in dataloaders['test']:
            input_features = input_features.to(DEVICE)
            targets = targets.to(DEVICE)
            
            loss = trained_model.forward_loss(input_features, targets)
            test_loss += loss.item()
        
    print(f'Final Test Loss: {test_loss / len(dataloaders["test"]):.4f}')

if __name__ == '__main__':
    main()