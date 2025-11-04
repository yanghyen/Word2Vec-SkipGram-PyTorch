import torch
import torch.nn as nn
import numpy as np
from nltk.corpus import brown

sentences = brown.sents()
text = [w for sent in sentences for w in sent]
vocab = set(text)
word2idx = {w: i for i, w in enumerate(vocab)}


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context_idx):
        embeds = self.embeddings(context_idxs)
        avg_embeds = embeds.mean(dim=1)
        out = self.linear(avg_embeds)
        return out 

vocab_size  = len(word2idx)
embedding_dim = 10

model = CBOW(vocab_size, embedding_dim)
model.load_state_dict(torch.load("CBOWbyPyTorch.pth"))
model.eval()    

def cosine_similarity(vec1, vec2):
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    return np.dot(vec1, vec2)

money_vec = model.embeddings(torch.tensor([word2idx["money"]])).detach().numpy()[0]
finance_vec = model.embeddings(torch.tensor([word2idx["finance"]])).detach().numpy()[0]

sim = cosine_similarity(money_vec, finance_vec)
print("money ~ finance 유사도:", sim)
