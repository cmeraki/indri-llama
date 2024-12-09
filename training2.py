import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
import numpy as np
from llama_model import get_model
import gc
import os
import time
import logging
from pathlib import Path
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ddp = int(os.environ.get('RANK', -1)) != -1
ddp_local_rank = int(os.environ.get('LOCAL_RANK', 0))
ddp_world_size = int(os.environ.get('WORLD_SIZE', 1))
device = torch.device(f'cuda:{ddp_local_rank}' if torch.cuda.is_available() else 'cpu')
master_process = ddp_local_rank == 0

class TrainingConfig:
    def __init__(
        self, 
        batch_size=16,  
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
    
    all_tokens = torch.tensor(all_tokens, dtype=torch.long, device='cpu')
    
    total_size = len(all_tokens)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - test_size - val_size
    
    torch.manual_seed(42)
    indices = torch.randperm(total_size)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    return {
        'train': AudioTokenDataset(all_tokens[train_indices]),
        'val': AudioTokenDataset(all_tokens[val_indices]),
        'test': AudioTokenDataset(all_tokens[test_indices])
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

def get_ctx(device_type):
    return torch.amp.autocast(device_type=device_type, dtype=torch.float16)

def get_lr(it, warmup_iters=2000, lr_decay_iters=600000, min_lr=1e-5, learning_rate=1e-3):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    
    if it > lr_decay_iters:
        return min_lr
    
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

def estimate_loss(model, ctx, eval_batches):
    model.eval()
    losses = {}
    with torch.no_grad():
        for name, batches in [('val', eval_batches)]:
            batch_losses = []
            for X, Y in batches:
                with ctx:
                    _, loss = model(X, Y)
                batch_losses.append(loss.item())
            losses[name] = np.mean(batch_losses)
    model.train()
    return losses

def train(
    model,
    dataloaders,
    out_dir='./checkpoints',
    steps=350000,
    batch_size=2,
    block_size=1024,
    grad_accum_steps=32,
    eval_interval=5000,
    eval_steps=100
):
    os.makedirs(out_dir, exist_ok=True)
    print(f"Moving model to {device}")
    model.to(device)

    if ddp:
        assert grad_accum_steps % ddp_world_size == 0
        grad_accum_steps //= ddp_world_size

    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    ctx = get_ctx(device_type)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=1e-3,
        betas=(0.9, 0.95),
        device_type=device_type
    )

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    eval_batches = [(X.to(device), Y.to(device)) 
                    for X, Y in list(zip(
                        dataloaders['val'].dataset.data[:eval_steps], 
                        dataloaders['val'].dataset.data[1:eval_steps+1]))]

    local_iter_num = 0
    t0 = time.time()
    all_losses = {'train': 0, 'val': 0}

    pbar = tqdm(total=steps, disable=not master_process)

    for iter_num in range(steps):
        for param_group in optimizer.param_groups:
            param_group['lr'] = get_lr(iter_num)

        if iter_num % eval_interval == 0 and master_process:
            losses = estimate_loss(model, ctx, eval_batches)
            logger.info(f"Validation loss: {iter_num}, {losses}")
            all_losses['val'] = losses.get('val', 0)
            
            raw_model = model.module if ddp else model
            model_fname = os.path.join(out_dir, f'llama_{iter_num}.pt')
            torch.save({
                "model": raw_model.state_dict(), 
                "config": raw_model.config
            }, model_fname)

        X, Y = next(iter(dataloaders['train']))
        X, Y = X.to(device), Y.to(device)

        for micro_step in range(grad_accum_steps):
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / grad_accum_steps
            
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        if master_process:
            lossf = loss.item() * grad_accum_steps
            all_losses['train'] = lossf
            loss_string = f"train: {all_losses['train']:.4f} val: {all_losses['val']:.4f}"
            pbar.set_description(loss_string)
            pbar.update(1)

        iter_num += 1
        local_iter_num += 1

    raw_model = model.module if ddp else model
    final_model_fname = os.path.join(out_dir, 'llama_last.pt')
    torch.save({
        "model": raw_model.state_dict(), 
        "config": raw_model.config
    }, final_model_fname)

    if ddp:
        dist.destroy_process_group()

    return final_model_fname

def main():
    config = TrainingConfig(
        batch_size=2, 
        grad_accumulation_steps=64,
        learning_rate=1e-4,
        epochs=1000
    )
    
    model = get_model(
        model_type='llama',
        path='Llama_tokenizer/model.safetensors',
        device=str(device),
        audio_feature_dim=128  
    )
    model.to(device)
    model.gradient_checkpointing_enable()

    datasets = prepare_data('tokens/lj_speech_tokens.pkl')
    
    dataloaders = {
        'train': DataLoader(
            datasets['train'], 
            batch_size=config.batch_size, 
            shuffle=True, 
            collate_fn=create_batch_collate(),
            pin_memory=True, 
            num_workers=2     
        ),
        'val': DataLoader(
            datasets['val'], 
            batch_size=config.batch_size, 
            shuffle=False, 
            collate_fn=create_batch_collate(),
            pin_memory=True
        )
    }

    trained_model = train(
        model, 
        dataloaders, 
        out_dir='./checkpoints',
        steps=350000,
        batch_size=config.batch_size,
        grad_accum_steps=config.grad_accumulation_steps
    )

    trained_model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for input_features, targets in dataloaders['test']:
            input_features = input_features.to(device)
            targets = targets.to(device)
            
            loss = trained_model.forward_loss(input_features, targets)
            test_loss += loss.item()
        
    print(f'Final Test Loss: {test_loss / len(dataloaders["test"]):.4f}')


if __name__ == '__main__':
    main()