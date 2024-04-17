import torch

def load_data(file):
    global text, vocab_size, stoi, itos
    with open(file, 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }

def get_train_data():
    global train_data, val_data
    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data

def get_batch(split, block_size, batch_size, device='cpu'):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l): 
    if isinstance(l, int):
        return itos[l]
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
