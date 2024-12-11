import os
import time
import math
import json
import torch
import random
import numpy as np
from pathlib import Path
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import Dataset, DataLoader
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.utils.checkpoint import checkpoint

class TokenDataset(Dataset):
    def __init__(self, tokens_file, block_size=2048):
        self.tokens = torch.load(tokens_file, map_location='cpu')
        self.block_size = block_size
        
        self.TEXT_START_TOKEN = 128000  
        self.TASK_TOKEN = 144642
        self.SPEAKER_TOKEN = 144645
        self.AUDIO_START_TOKEN = 144641
        self.STOP_TOKEN = 144644

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        full_sequence = self.tokens[idx]
        
        audio_start_idx = torch.where(full_sequence == self.AUDIO_START_TOKEN)[0]
        
        if len(audio_start_idx) == 0:
            raise ValueError("Audio start token not found in the sequence")
        
        audio_start_idx = audio_start_idx[0]
        
        if full_sequence.size(0) > self.block_size:
            full_sequence = full_sequence[:self.block_size]
        
        x = full_sequence.clone()
        y = full_sequence.clone()
        
        metadata = {
            'text_tokens': full_sequence[:audio_start_idx].tolist(),
            'audio_tokens': full_sequence[audio_start_idx+1:].tolist(),
            'task_token_pos': torch.where(full_sequence == self.TASK_TOKEN)[0][0] if self.TASK_TOKEN in full_sequence else -1,
            'speaker_token_pos': torch.where(full_sequence == self.SPEAKER_TOKEN)[0][0] if self.SPEAKER_TOKEN in full_sequence else -1
        }
        
        return x, y, metadata

def get_lr(it, warmup_iters=2000, lr_decay_iters=300000, min_lr=1e-6, learning_rate=6e-4):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

class MultimodalTrainer:
    def __init__(self, 
                 model, 
                 tokens_file, 
                 out_dir='./out', 
                 batch_size=2,  
                 block_size=2048, 
                 learning_rate=6e-4,
                 grad_accum_steps=128,  
                 steps=100000):
        
        self.ddp = int(os.environ.get('RANK', -1)) != -1
        
        if self.ddp:
            init_process_group(backend='nccl')
            self.ddp_rank = int(os.environ['RANK'])
            self.ddp_local_rank = int(os.environ['LOCAL_RANK'])
            self.ddp_world_size = int(os.environ['WORLD_SIZE'])
            self.device = f'cuda:{self.ddp_local_rank}'
            torch.cuda.set_device(self.device)
            self.master_process = self.ddp_rank == 0
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.master_process = True
            self.ddp_world_size = 1
        
        self.model = model.to(self.device)
        
        self.model.enable_gradient_checkpointing()
        
        if self.ddp:
            self.model = DDP(self.model, device_ids=[self.ddp_local_rank], 
                              gradient_as_bucket_view=True,  
                              find_unused_parameters=False)
        
        self.dataset = TokenDataset(tokens_file, block_size=block_size)
        
        self.batch_size = batch_size
        self.block_size = block_size
        self.steps = steps
        self.out_dir = out_dir
        self.learning_rate = learning_rate
        self.grad_accum_steps = grad_accum_steps
        
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=4, 
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        
        self.optimizer = ZeroRedundancyOptimizer(
            self.model.parameters(),
            optimizer_class=torch.optim.AdamW,
            lr=learning_rate,
            weight_decay=0.1,
            betas=(0.9, 0.95)
        )
        
        self.scaler = torch.cuda.amp.GradScaler(
            init_scale=2.**16,  
            growth_factor=2.0,  
            backoff_factor=0.5,  
            growth_interval=2000  
        )
        
        os.makedirs(out_dir, exist_ok=True)

    def log_memory_usage(self):
        """Log current GPU memory usage"""
        if torch.cuda.is_available():
            print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    def train(self):
        iter_num = 0
        best_val_loss = float('inf')
        
        if self.master_process:
            print("Starting training with memory optimizations...")
        
        while iter_num < self.steps:
            for batch_x, batch_y, batch_metadata in self.dataloader:
                # Dynamic learning rate
                lr = get_lr(iter_num)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    logits, loss = self.model(batch_x, batch_y)
                
                loss = loss / self.grad_accum_steps
                
                self.scaler.scale(loss).backward()
                
                if (iter_num + 1) % self.grad_accum_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                
                if self.master_process and iter_num % 500 == 0:
                    print(f"Step {iter_num}, Loss: {loss.item()}")
                    self.log_memory_usage()
                    
                    checkpoint = {
                        'model': self.model.module.state_dict() if self.ddp else self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'iter': iter_num,
                        'loss': loss.item()
                    }
                    torch.save(checkpoint, f"{self.out_dir}/checkpoint_{iter_num}.pt")
                
                iter_num += 1
                
                if iter_num >= self.steps:
                    break
        
        if self.master_process:
            final_model_path = f"{self.out_dir}/final_model.pt"
            torch.save({
                'model': self.model.module.state_dict() if self.ddp else self.model.state_dict(),
                'config': self.model.module.config if self.ddp else self.model.config
            }, final_model_path)
            print(f"Training completed. Final model saved to {final_model_path}")
        
        if self.ddp:
            destroy_process_group()

def main():
    from llama_model import get_model 

    tokens_file = 'tokens/lj_speech_tokens.pkl'  
    vocab_size = 144645  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_model(
        model_type='llama-1b',
        vocab_size=vocab_size,
        dropout=0.1,
        max_seq_len=2048,
        bias=False,
        device=device
    )

    def enable_gradient_checkpointing(self):
        for block in self.blocks:
            block.attn = checkpoint(block.attn)
            block.mlp = checkpoint(block.mlp)
    
    model.enable_gradient_checkpointing = enable_gradient_checkpointing.__get__(model)

    trainer = MultimodalTrainer(
        model=model, 
        tokens_file=tokens_file, 
        batch_size=2,  
        block_size=2048, 
        steps=100000,
        grad_accum_steps=128  
    )
    
    trainer.train()

if __name__ == '__main__':
    main()