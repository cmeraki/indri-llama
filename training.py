import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.utils.data import random_split
import pickle
from llama_model import Llama, LlamaConfig
import torch.nn.functional as F
from tqdm import tqdm

class TokenDataset(Dataset):
    def __init__(self, tokens_file, max_token=144645, max_sequence_length=1024):
        with open(tokens_file, 'rb') as f:
            all_tokens = pickle.load(f)
        
        self.tokens = [
            seq[:max_sequence_length].tolist() 
            for seq in all_tokens 
            if len(seq) > 0
        ]
        
        print(f"Total sequences: {len(self.tokens)}")
        print(f"Max sequence length: {max(len(seq) for seq in self.tokens)}")

    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, idx):
        sequence = self.tokens[idx]
        
        if 144641 in sequence:
            split_idx = sequence.index(144641) + 1
        else:
            split_idx = len(sequence)
        
        input_seq = sequence[:split_idx]
        output_seq = sequence[split_idx:]
        
        input_seq = torch.tensor(input_seq, dtype=torch.long)
        output_seq = torch.tensor(output_seq, dtype=torch.long)
        
        return input_seq, output_seq

def collate_fn(batch):
    inputs, targets = zip(*batch)    
    max_input_len = max(len(inp) for inp in inputs)
    max_target_len = max(len(tgt) for tgt in targets)
    max_len = min(max_input_len, max_target_len)      
    
    inputs_padded = torch.stack([
        F.pad(inp[:max_len], (0, max_len - len(inp[:max_len])), value=0) 
        for inp in inputs
    ])
    
    targets_padded = torch.stack([
        F.pad(tgt[:max_len], (0, max_len - len(tgt[:max_len])), value=-100)  
        for tgt in targets
    ])

    attention_mask = (inputs_padded != 0).float()
    
    return inputs_padded, targets_padded, attention_mask

def get_vocab_size(tokens_file):
    with open(tokens_file, 'rb') as f:
        tokens = pickle.load(f)
    
    max_token_in_data = max(max(seq) for seq in tokens)
    return max_token_in_data + 1

def train_model(
    rank, 
    world_size, 
    model, 
    dataloader, 
    optimizer, 
    num_epochs=1000, 
    save_interval=100, 
    gradient_accumulation_steps=4
):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    device = torch.device(f'cuda:{rank}')
    
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    
    model.train()
    
    for epoch in range(num_epochs):
        if hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)
        
        total_loss = 0
        optimizer.zero_grad()
        
        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=rank!=0)):
            inputs, targets, attention_mask = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            attention_mask = attention_mask.to(device)
        
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                loss = model.module.forward_loss(inputs, targets, attention_mask=attention_mask)
            
            loss = loss / gradient_accumulation_steps
            
            loss.backward()
            
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item()
        
        if gradient_accumulation_steps > 1:
            optimizer.step()
            optimizer.zero_grad()
        
        if rank == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")        
            
            if (epoch + 1) % save_interval == 0 or (epoch + 1) == num_epochs:
                torch.save(model.module.state_dict(), f"llama_model_epoch_{epoch+1}.pth")
    
    dist.destroy_process_group()

def main(num_gpus=2):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    tokens_file = 'tokens/lj_speech_tokens.pkl'
    max_token = 144645   
    vocab_size = get_vocab_size(tokens_file)
    print(f"Detected Vocabulary Size: {vocab_size}")
    
    config = LlamaConfig(
        dim=1024,  
        n_layers=12,  
        n_heads=16,  
        vocab_size=vocab_size,  
        max_seq_len=1024,  
        multiple_of=256,
        use_scaled_rope=True
    )
    
    dataset = TokenDataset(tokens_file, max_token, max_sequence_length=1024)
    
    mp.spawn(
        run_distributed_training, 
        args=(num_gpus, dataset, config), 
        nprocs=num_gpus
    )

def run_distributed_training(rank, num_gpus, dataset, config):
    dataset_size = len(dataset)
    train_size = int(0.9 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=num_gpus, 
        rank=rank,
        shuffle=True
    )
    
    dataloader = DataLoader(
        train_dataset, 
        batch_size=48 // num_gpus,  
        shuffle=False, 
        sampler=train_sampler,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=2,
        prefetch_factor=2
    )
    
    torch.manual_seed(42 + rank)
    model = Llama(config).to(rank)
    
    pre_trained_weights = ''
    try:
        state_dict = torch.load(pre_trained_weights, map_location=f'cuda:{rank}', weights_only=True)
        model.load_state_dict(state_dict)
        if rank == 0:
            print("Loaded pre-trained weights.")
    except FileNotFoundError:
        if rank == 0:
            print("Pre-trained weights not found, training from scratch.")
    
    optimizer = model.configure_optimizers(
        weight_decay=0.01, 
        learning_rate=1e-5,  
        betas=(0.9, 0.95), 
        device_type='cuda'
    )
    
    torch.autograd.set_detect_anomaly(True)
    
    train_model(
        rank, 
        num_gpus, 
        model, 
        dataloader, 
        optimizer, 
        num_epochs=1000, 
        save_interval=100,
        gradient_accumulation_steps=8
    )

if __name__ == "__main__":
    main(num_gpus=2)