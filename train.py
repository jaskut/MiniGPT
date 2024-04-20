# dataset: https://www.kaggle.com/datasets/louise2001/goethe/data

from tqdm import tqdm
import torch
from GPTLanguageModel import GPTLanguageModel
import config
import data

torch.manual_seed(1337)

load_model = False

# hyperparameters
max_iters = 1000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 100

default_params = {
  'batch_size': 64, # how many independent sequences will we process in parallel?
  'block_size': 256, # what is the maximum context length for predictions?
  'n_embd': 384,
  'n_head': 6,
  'n_layer': 6,
  'dropout': 0.2,
}


print(("Using gpu" if config.device != 'cpu' else "Using cpu"))

data.load_data('goethe/full.txt', word_tokens_param=True)
data.get_train_data()


@torch.no_grad()
def estimate_loss(model, block_size, batch_size):
  out = {}
  model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = data.get_batch(split, block_size=block_size, batch_size=batch_size, device=config.device)
      logits, loss = model(X,Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out


if __name__ == "__main__":
  params = default_params
  epoch = 0

  if load_model:
    checkpoint = torch.load(config.MODEL_PATH)
    params = checkpoint["params"]
    epoch = checkpoint["epoch"]

  model = GPTLanguageModel(data.vocab_size, device=config.device, **params)
  if load_model:
    model.load_state_dict(checkpoint["model_state_dict"])

  m = model.to(config.device) 
  # print the number of parameters in the model
  print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

  # create a PyTorch optimizer
  optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
  if load_model:
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

  for iter in range(epoch, epoch + max_iters//eval_interval):

    # every once in a while evaluate the loss on train and val sets
    losses = estimate_loss(m, params["block_size"], params["batch_size"])
    print(f"epoch {iter+1} of {epoch + max_iters//eval_interval}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    for i in tqdm(range(eval_interval)):
      # sample a batch of data
      xb, yb = data.get_batch('train', block_size=params["block_size"], batch_size=params["batch_size"], device=config.device)

      # evaluate the loss
      logits, loss = model(xb, yb)
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      optimizer.step()

    torch.save({
    'epoch': iter + 1,
    'model_state_dict': m.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'losses': losses, 
    'params': default_params,
    }, config.MODEL_PATH)

  losses = estimate_loss(m, params["block_size"], params["batch_size"])
  print(f"step {max_iters} of {max_iters}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
  # generate from the model
  m.eval()
  context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
  print(data.decode(m.generate(context, max_new_tokens=500, block_size=params["block_size"])[0].tolist()))
  #open(f'werke/werk_{iter+1}.txt', 'w').write(data.decode(m.generate(context, max_new_tokens=10000, block_size=params["block_size"])[0].tolist()))