# source: https://github.com/karpathy/ng-video-lecture/tree/master
# video: https://www.youtube.com/watch?v=kCc8FmEb1nY
# dataset: https://www.kaggle.com/datasets/louise2001/goethe/data

from tqdm import tqdm
import torch
from GPTLanguageModel import GPTLanguageModel

torch.manual_seed(1337)

SAVE_TO_PATH = 'model_weights.pth'

# hyperparameters
max_iters = 500
eval_interval = 100
learning_rate = 3e-4
eval_iters = 200

batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
n_embd = 32
n_head = 4
n_layer = 3
dropout = 0.2


if torch.cuda.is_available():
    device = 'cuda'
if torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
print(("Using GPU" if device != 'cpu' else "Using CPU"))

with open('goethe/full.txt', 'r', encoding='utf-8') as f:
  text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = get_batch(split)
      logits, loss = model(X,Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out


if __name__ == "__main__":
  model = GPTLanguageModel(vocab_size, batch_size=batch_size, block_size=block_size, n_embd=n_embd, n_head=n_head, n_layer=n_layer, dropout=dropout, device=device)
  m = model.to(device)
  # print the number of parameters in the model
  print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

  # create a PyTorch optimizer
  optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

  for iter in range(max_iters//eval_interval):

    # every once in a while evaluate the loss on train and val sets
    losses = estimate_loss()
    print(f"step {iter*eval_interval} of {max_iters}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    for i in tqdm(range(eval_interval)):
      # sample a batch of data
      xb, yb = get_batch('train')

      # evaluate the loss
      logits, loss = model(xb, yb)
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      optimizer.step()

  losses = estimate_loss()
  print(f"step {max_iters} of {max_iters}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
  torch.save(m.state_dict(), SAVE_TO_PATH)
  # generate from the model
  context = torch.zeros((1, 1), dtype=torch.long, device=device)
  print(decode(m.generate(context, max_new_tokens=500, block_size=block_size)[0].tolist()))
  #print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))