import torch

MODEL_PATH = 'models/model_1.pt'
word_tokens = False

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
    