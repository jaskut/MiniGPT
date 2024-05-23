import torch
from GPTLanguageModel import GPTLanguageModel
import data
import config

data.load_data('goethe/full.txt', word_tokens_param=config.word_tokens)

checkpoint = torch.load(config.MODEL_PATH, map_location='cpu')
params = checkpoint["params"]

model = GPTLanguageModel(data.vocab_size, device=config.device, **params)
model.load_state_dict(checkpoint["model_state_dict"])
m = model.to(config.device) 
m.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
for _ in range(500):
    context = m.generate(context, max_new_tokens=1, block_size=params["block_size"])
    print(data.decode(context[0].tolist()[-1]), end='', flush=True)
open(f'werke/werk_create.txt', 'w').write(data.decode(m.generate(context, max_new_tokens=2000, block_size=params["block_size"], skip=config.word_tokens)[0].tolist()))