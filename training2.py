import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import pickle
from sklearn.model_selection import train_test_split
from llama_model import Llama, LlamaConfig  

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
        return self.tokens[idx]

def train_model(model, train_loader, optimizer, criterion, device, num_epochs=10):
    model.train()
    scaler = torch.cuda.amp.GradScaler()  
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():  
                outputs = model.forward_loss(batch[:, :-1], batch[:, 1:])
                loss = criterion(outputs, batch[:, 1:])
            scaler.scale(loss).backward()  
            scaler.step(optimizer)  
            scaler.update()  
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

def main():
    tokens = load_data('tokens/lj_speech_tokens.pkl')  
    train_tokens, test_tokens = train_test_split(tokens, test_size=0.2, random_state=42)
    val_tokens, test_tokens = train_test_split(test_tokens, test_size=0.5, random_state=42)

    train_dataset = TokenDataset(train_tokens)
    val_dataset = TokenDataset(val_tokens)
    test_dataset = TokenDataset(test_tokens)

    train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    config = LlamaConfig(vocab_size=144448, dim=1024, n_layers=12, n_heads=16)  
    model = Llama(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-100) 

    train_model(model, train_loader, optimizer, criterion, device, num_epochs=10)

    torch.save(model.state_dict(), 'llama_model.pth')

if __name__ == '__main__':
    main()