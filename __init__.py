import torch
from .GPTLanguageModel import GPTLanguageModel
from . import data
from . import config

class MiniGPT(GPTLanguageModel):
  def __init__(self, from_path=config.MODEL_PATH, seed=None):
    if seed:
      torch.manual_seed(seed)

    checkpoint = torch.load(from_path, map_location='cpu')
    self.params = checkpoint["params"]
    data.load_vocab(checkpoint["vocab"], word_tokens_param=self.params["word_tokens"])

    super().__init__(data.vocab_size, device=config.device, **self.params)
    self.load_state_dict(checkpoint["model_state_dict"])
    self.to(config.device) 
    self.eval()

  def create(self, prompt, max_tokens=100, n=1):
    prompt_encoded = torch.tensor(data.encode(prompt), dtype=torch.int)
    context = torch.stack([prompt_encoded for _ in range(n)]).to(config.device)

    context = self.generate(context, max_new_tokens=max_tokens-len(prompt_encoded), block_size=self.params["block_size"])
    return [{"text": data.decode(option.tolist())} for option in context]