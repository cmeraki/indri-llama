import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
from llama_model import Llama, LlamaConfig
import torch.nn.functional as F
from tqdm import tqdm

class TokenDataset(Dataset):
    def __init__(self, tokens_file, max_token=144641):
        with open(tokens_file, 'rb') as f:
            all_tokens = pickle.load(f)
        
        self.tokens = [
            seq.tolist() for seq in all_tokens
        ]
        
        print(f"Total sequences: {len(self.tokens)}")
        print(f"Max sequence length after filtering: {max(len(seq) for seq in self.tokens)}")

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

def train_model(model, dataloader, optimizer, num_epochs=1000, save_interval=1000):
    model.train()
    device = next(model.parameters()).device
  
    for epoch in range(num_epochs):
        total_loss = 0  
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, targets, attention_mask = batch
            inputs, targets, attention_mask = inputs.to(device), targets.to(device), attention_mask.to(device)
        
            optimizer.zero_grad()
            loss = model.forward_loss(inputs, targets, attention_mask=attention_mask)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")        
        
        if (epoch + 1) % save_interval == 0 or (epoch + 1) == num_epochs:
            torch.save(model.state_dict(), f"llama_model_epoch_{epoch+1}.pth")

def main():
    tokens_file = 'tokens/lj_speech_tokens.pkl'
    max_token = 144641  
    
    vocab_size = get_vocab_size(tokens_file)
    print(f"Detected Vocabulary Size: {vocab_size}")
    
    config = LlamaConfig(
        dim=1024,  
        n_layers=12,
        n_heads=16,
        vocab_size=vocab_size,  
        max_seq_len=2048,  
        multiple_of=2048,
        use_scaled_rope=True
    )
    
    dataset = TokenDataset(tokens_file, max_token)
    dataloader = DataLoader(
        dataset, 
        batch_size=16,
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    torch.manual_seed(42)  
    model = Llama(config).cuda()
    
    pre_trained_weights = 'path_to_pretrained_weights.pth'
    try:
        model.load_state_dict(torch.load(pre_trained_weights))
        print("Loaded pre-trained weights.")
    except FileNotFoundError:
        print("Pre-trained weights not found, training from scratch.")
    
    optimizer = model.configure_optimizers(
        weight_decay=0.01, 
        learning_rate=1e-4, 
        betas=(0.9, 0.95), 
        device_type='cuda'
    )
    
    torch.autograd.set_detect_anomaly(True)
    
    train_model(model, dataloader, optimizer, num_epochs=10, save_interval=10)

if __name__ == "__main__":
    main()