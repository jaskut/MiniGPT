import torch
import re

def load_data(file, word_tokens_param=False):
    global text, vocab_size, stoi, itos, word_tokens
    word_tokens = word_tokens_param
    with open(file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    if word_tokens:
        from torchtext.vocab import build_vocab_from_iterator
        text_stripped= re.sub(r'[\W|\d|_]', ' ', text).lower()
        text= re.sub(r'\b[a-zÀ-ÿ]\b', '', text_stripped)
        vocab = build_vocab_from_iterator([text.split()], max_tokens=5000, specials=["<unk>"])
        vocab.set_default_index(vocab["<unk>"])
        vocab_size = len(vocab)
        stoi = vocab
        itos = vocab.get_itos()

    else:
        chars = sorted(list(set(text)))
        vocab_size = len(chars)

        stoi = { ch:i for i,ch in enumerate(chars) }
        itos = chars

def get_train_data():
    global train_data, val_data
    # Train and test splits
    if word_tokens:
        data = torch.tensor(encode(text.split()), dtype=torch.long)
    else:
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
    if word_tokens:
        return ' '.join([itos[i] for i in l])
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
