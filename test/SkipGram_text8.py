import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import random
import requests, zipfile, io
import nltk

nltk.download("punkt")

# ----------------------
# 1. Load Text8 Dataset
# ----------------------
url = "http://mattmahoney.net/dc/text8.zip"
print("Downloading Text8 dataset...")
r = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(r.content))
text8 = z.read("text8").decode("utf-8")
print("Text8 loaded!")

tokens = text8.split()
print("Total tokens:", len(tokens))

# ----------------------
# 2. Build Vocab
# ----------------------
vocab_size = 30000  # 제한
counts = Counter(tokens)
vocab = [w for w, _ in counts.most_common(vocab_size - 1)]
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}

UNK = len(word2idx)
word2idx["<UNK>"] = UNK
idx2word[UNK] = "<UNK>"

corpus = [word2idx.get(w, UNK) for w in tokens]

# ----------------------
# 3. Subsampling
# ----------------------
total_count = len(corpus)
freqs = {w: c / total_count for w, c in counts.items()}
t = 1e-5
subsampled = [
    w for w in corpus if random.random() < (1 - (t / freqs.get(idx2word[w], 1))**0.5)
]
print("After subsampling:", len(subsampled))

# ----------------------
# 4. Generate Skip-gram (center, context) pairs
# ----------------------
window_size = 2
pairs = []
for i, center in enumerate(subsampled):
    for j in range(-window_size, window_size + 1):
        if j == 0 or i + j < 0 or i + j >= len(subsampled):
            continue
        context = subsampled[i + j]
        pairs.append((center, context))
print("Total skip-gram pairs:", len(pairs))

# ----------------------
# 5. Dataset + Dataloader
# ----------------------
class SkipGramDataset(torch.utils.data.Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        return torch.tensor(self.pairs[idx][0]), torch.tensor(self.pairs[idx][1])

dataset = SkipGramDataset(pairs)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)

# ----------------------
# 6. Skip-gram Model (Negative Sampling)
# ----------------------
class SkipGramNeg(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_dim)
        self.out_embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, center, pos_context, neg_context):
        center_emb = self.in_embed(center)                # (batch, embed)
        pos_emb = self.out_embed(pos_context)             # (batch, embed)
        neg_emb = self.out_embed(neg_context)             # (batch, K, embed)

        pos_score = torch.mul(center_emb, pos_emb).sum(dim=1)
        pos_loss = torch.log(torch.sigmoid(pos_score))

        neg_score = torch.bmm(neg_emb, center_emb.unsqueeze(2)).squeeze()
        neg_loss = torch.log(torch.sigmoid(-neg_score)).sum(dim=1)

        return -(pos_loss + neg_loss).mean()

# ----------------------
# 7. Training
# ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

embed_dim = 100
model = SkipGramNeg(vocab_size, embed_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

def get_negative_samples(batch_size, K=5):
    """샘플링 분포에 따라 negative words 뽑기"""
    negs = torch.randint(0, vocab_size, (batch_size, K), device=device)
    return negs

for epoch in range(20):
    total_loss = 0
    for center, context in dataloader:
        center, context = center.to(device), context.to(device)
        neg_context = get_negative_samples(center.size(0), K=5)

        loss = model(center, context, neg_context)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# ----------------------
# 8. Test (word similarity)
# ----------------------
def most_similar(word, topn=5):
    if word not in word2idx:
        return []
    idx = word2idx[word]
    emb = model.in_embed.weight.detach().cpu()
    query = emb[idx]
    cos_sim = torch.mv(emb, query) / (
        torch.norm(emb, dim=1) * torch.norm(query) + 1e-9
    )
    sim_idx = torch.topk(cos_sim, topn + 1).indices.tolist()
    return [idx2word[i] for i in sim_idx if i != idx][:topn]

print("Most similar to 'king':", most_similar("king"))
print("Most similar to 'bank':", most_similar("bank"))

